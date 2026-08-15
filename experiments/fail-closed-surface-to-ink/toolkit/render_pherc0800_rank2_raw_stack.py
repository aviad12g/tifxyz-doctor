#!/usr/bin/env python3
"""Fail-closed full-resolution raw render for the repaired PHerc0800 candidate.

The stored-grid candidate is fixed upstream.  This program first resamples that
exact explicit mask/coordinate surface with the canonical linear TIFFXYZ
loader, then recomputes full-raster topology and native geometry through
normal offsets -46..+46.  It does not contact the raw volume unless every
preflight gate passes.  A passing surface is rendered from the official public
raw Zarr as one lossless uint8 positive-normal stack; the symmetric offset
range makes both normal signs and both layer orders exact index reversals.

No ink model, detector, OCR, or paid compute is used here.  Diagnostic PNGs are
only visibility aids derived deterministically from the raw stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage

from native_surface_sampler import SurfaceGrid, validate_surface_geometry
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import (
    ZarrV2ArraySpec,
    assemble_roi,
    chunk_key,
    decode_chunk_payload,
    fetch_bytes,
    fetch_json,
    sha256_bytes,
    sha256_file,
)
from tifxyz_render_pipeline import (
    RenderOptions,
    load_tifxyz_asset,
    render_surface_to_directory,
)


CANDIDATE_ID = "20251028220042-auto_grown_20251028220042762"
DEFAULT_SURFACE = Path(
    "outputs/first-letters-geometry/"
    "pherc0800-rank2-masked-validation-v2-topology-repaired/"
    "materialized-topology-repaired-component"
)
DEFAULT_VALIDATION = Path(
    "outputs/first-letters-geometry/"
    "pherc0800-rank2-masked-validation-v2-topology-repaired/validation.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/pherc0800-rank2-raw-stack-93"
)
DEFAULT_RAW_ROOT = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc0800/volumes/"
    "20250521135224-8.640um-1.2m-116keV-masked.zarr"
)
DEFAULT_ARRAY_PATH = "0"
PINNED_RAW_SHAPE_ZYX = (24298, 9867, 9867)
VOXEL_UM = 8.64
OFFSET_VOXELS = tuple(range(-46, 47))
MINIMUM_AREA_CM2 = 0.5
RAW_USER_AGENT = "pherc0800-first-letters-raw-render/1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def active_quads(vertex_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    mask = np.asarray(vertex_mask, dtype=bool)
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def quad_union_vertex_mask(quad_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    quads = np.asarray(quad_mask, dtype=bool)
    mask = np.zeros((quads.shape[0] + 1, quads.shape[1] + 1), dtype=bool)
    mask[:-1, :-1] |= quads
    mask[:-1, 1:] |= quads
    mask[1:, :-1] |= quads
    mask[1:, 1:] |= quads
    return mask


def scalable_topology_report(vertex_mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Vectorized topology of a potentially multi-million-pixel quad mask."""

    mask = np.asarray(vertex_mask, dtype=bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError("vertex mask must be a 2-D grid with both dimensions >=2")
    quads = active_quads(mask)
    used = quad_union_vertex_mask(quads)

    # Each grid edge has at most two incident quads.  XOR identifies boundary
    # edges without constructing millions of Python tuples.
    horizontal_incidence = np.zeros((mask.shape[0], mask.shape[1] - 1), dtype=np.uint8)
    horizontal_incidence[:-1] += quads
    horizontal_incidence[1:] += quads
    vertical_incidence = np.zeros((mask.shape[0] - 1, mask.shape[1]), dtype=np.uint8)
    vertical_incidence[:, :-1] += quads
    vertical_incidence[:, 1:] += quads
    boundary_horizontal = horizontal_incidence == 1
    boundary_vertical = vertical_incidence == 1
    boundary_degree = np.zeros(mask.shape, dtype=np.uint8)
    boundary_degree[:, :-1] += boundary_horizontal
    boundary_degree[:, 1:] += boundary_horizontal
    boundary_degree[:-1] += boundary_vertical
    boundary_degree[1:] += boundary_vertical
    boundary_vertices = boundary_degree > 0

    connectivity = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
    )
    _quad_labels, component_count = ndimage.label(quads, structure=connectivity)
    _boundary_labels, boundary_components = ndimage.label(
        boundary_vertices, structure=connectivity
    )
    degree_values, degree_counts = np.unique(
        boundary_degree[boundary_vertices], return_counts=True
    )
    degree_histogram = {
        str(int(degree)): int(count)
        for degree, count in zip(degree_values, degree_counts, strict=True)
    }
    q_padded = np.pad(quads, ((1, 1), (1, 1)), constant_values=False)
    nw = q_padded[:-1, :-1]
    ne = q_padded[:-1, 1:]
    sw = q_padded[1:, :-1]
    se = q_padded[1:, 1:]
    bowties = (nw & se & ~ne & ~sw) | (ne & sw & ~nw & ~se)
    bowtie_coordinates = np.argwhere(bowties)

    vertices = int(np.count_nonzero(used))
    edges = int(
        np.count_nonzero(horizontal_incidence)
        + np.count_nonzero(vertical_incidence)
    )
    faces = int(np.count_nonzero(quads))
    euler = vertices - edges + faces
    manifold_boundary = bool(
        np.count_nonzero(boundary_vertices)
        and set(degree_histogram) == {"2"}
        and not np.any(horizontal_incidence > 2)
        and not np.any(vertical_incidence > 2)
        and bowtie_coordinates.size == 0
    )
    holes = int(component_count) - euler if manifold_boundary else None
    expected_loops = int(component_count) + holes if holes is not None else None
    reasons: list[str] = []
    if int(component_count) != 1:
        reasons.append("quad complex is not one connected component")
    if not np.array_equal(used, mask):
        reasons.append("vertex mask contains vertices unused by any retained quad")
    if set(degree_histogram) != {"2"}:
        reasons.append("boundary graph contains open ends or branch/pinch vertices")
    if bowtie_coordinates.size:
        reasons.append("quad complex contains diagonal bowtie/pinch vertices")
    if holes is not None and holes < 0:
        reasons.append("Euler-derived hole count is negative")
    if expected_loops is not None and int(boundary_components) != expected_loops:
        reasons.append("boundary-loop count disagrees with Euler characteristic")
    return {
        "complex": "full-render-grid all-four-valid axis-aligned quads",
        "vertex_mask_count": int(np.count_nonzero(mask)),
        "vertex_count_used_by_quads": vertices,
        "orphan_vertex_count": int(np.count_nonzero(mask & ~used)),
        "edge_count": edges,
        "quad_face_count": faces,
        "euler_characteristic": euler,
        "connected_quad_component_count": int(component_count),
        "boundary_edge_count": int(
            np.count_nonzero(boundary_horizontal)
            + np.count_nonzero(boundary_vertical)
        ),
        "boundary_graph_component_count": int(boundary_components),
        "boundary_degree_histogram": degree_histogram,
        "boundary_loops_if_manifold": int(boundary_components)
        if manifold_boundary
        else None,
        "holes_if_manifold": holes,
        "maximum_edge_incidence": int(
            max(
                np.max(horizontal_incidence, initial=0),
                np.max(vertical_incidence, initial=0),
            )
        ),
        "bowtie_vertex_count": int(bowtie_coordinates.shape[0]),
        "bowtie_vertices_row_column": bowtie_coordinates[:100].astype(int).tolist(),
        "bowtie_coordinate_list_truncated": bool(bowtie_coordinates.shape[0] > 100),
        "induced_quad_closure_exact": bool(np.array_equal(used, mask)),
        "nonmanifold_risk": bool(reasons),
        "nonmanifold_risk_reasons": reasons,
        "self_intersection_status": "unknown_not_computed",
    }


def physical_area_cm2(
    surface: SurfaceGrid, *, voxel_um: float = VOXEL_UM, row_chunk: int = 256
) -> float:
    """Sum full-resolution quad area with bounded temporary memory."""

    total_um2 = 0.0
    scale = float(voxel_um)
    for row0 in range(0, surface.shape[0] - 1, row_chunk):
        row1 = min(row0 + row_chunk, surface.shape[0] - 1)
        rows = slice(row0, row1 + 1)
        points = np.stack(
            (surface.x[rows], surface.y[rows], surface.z[rows]), axis=-1
        ) * scale
        valid = surface.valid[rows]
        quads = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        p00, p01 = points[:-1, :-1], points[:-1, 1:]
        p10, p11 = points[1:, :-1], points[1:, 1:]
        triangle_one = np.cross(p01 - p00, p10 - p00)
        triangle_two = np.cross(p11 - p10, p11 - p01)
        area = 0.5 * (
            np.linalg.norm(triangle_one, axis=-1)
            + np.linalg.norm(triangle_two, axis=-1)
        )
        total_um2 += float(np.sum(area[quads], dtype=np.float64))
    return total_um2 / 100_000_000.0


def compact_geometry(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "surface_shape",
            "volume_shape_zyx",
            "coordinate_min_xyz",
            "coordinate_max_xyz",
            "valid_vertex_count",
            "in_bounds_vertex_fraction",
            "normal_valid_vertex_fraction",
            "valid_neighbor_edge_count",
            "median_neighbor_distance_um",
            "p95_neighbor_distance_um",
            "max_neighbor_distance_um",
            "continuity_limit_um",
            "discontinuity_edge_count",
            "neighbor_wrap_limit_um",
            "neighbor_wrap_risk_edge_count",
            "valid_quad_count",
            "median_quad_area_um2",
            "degenerate_quad_count",
            "folded_quad_count",
            "metric_distortion_p95",
            "distorted_quad_count",
            "abrupt_normal_flip_edge_count",
            "offset_sample_in_bounds_fraction",
            "warnings",
        )
    }


def geometry_pass(report: Mapping[str, Any]) -> bool:
    return bool(
        float(report["in_bounds_vertex_fraction"]) == 1.0
        and float(report["normal_valid_vertex_fraction"]) == 1.0
        and float(report["offset_sample_in_bounds_fraction"]) == 1.0
        and all(
            int(report[name]) == 0
            for name in (
                "discontinuity_edge_count",
                "neighbor_wrap_risk_edge_count",
                "degenerate_quad_count",
                "folded_quad_count",
                "distorted_quad_count",
                "abrupt_normal_flip_edge_count",
            )
        )
    )


def full_resolution_preflight(
    surface_directory: Path,
    validation_path: Path,
    output_directory: Path,
    *,
    candidate_id: str = CANDIDATE_ID,
    expected_shape_yx: tuple[int, int] = (1860, 1900),
    resample_shape_yx: tuple[int, int] | None = None,
    voxel_um: float = VOXEL_UM,
    volume_shape_zyx: tuple[int, int, int] = PINNED_RAW_SHAPE_ZYX,
    accepted_validation_verdicts: tuple[str, ...] = ("pass_to_free_raw_render",),
) -> tuple[Any, dict[str, Any]]:
    validation_payload = validation_path.read_bytes()
    validation = json.loads(validation_payload)
    if validation.get("verdict") not in accepted_validation_verdicts:
        raise ValueError("stored-grid validation does not authorize a free raw render")
    initial_asset = load_tifxyz_asset(
        surface_directory,
        resolution="full",
        output_shape=resample_shape_yx,
        interpolation="linear",
        maximum_surface_pixels=10_000_000,
    )
    initial_mask = initial_asset.surface.valid
    initial_topology = scalable_topology_report(initial_mask)
    initial_quads = active_quads(initial_mask)
    render_mask = quad_union_vertex_mask(initial_quads)
    if np.any(render_mask & ~initial_mask):
        raise RuntimeError("full-resolution orphan cleanup added a mask pixel")
    if not np.array_equal(active_quads(render_mask), initial_quads):
        raise RuntimeError("full-resolution orphan cleanup changed the quad set")
    cleaned_surface = SurfaceGrid.from_tifxyz(
        initial_asset.surface.x,
        initial_asset.surface.y,
        initial_asset.surface.z,
        mask=render_mask,
    )
    asset = replace(
        initial_asset,
        surface=cleaned_surface,
        mask_source=(
            initial_asset.mask_source
            + "; strictly-subtractive-full-resolution-unused-vertex-cleanup"
        ),
    )
    topology = scalable_topology_report(asset.surface.valid)
    area_cm2 = physical_area_cm2(asset.surface, voxel_um=voxel_um)
    offsets_um = tuple(float(offset) * voxel_um for offset in OFFSET_VOXELS)
    reports: dict[str, dict[str, Any]] = {}
    for name, sign in (("positive", 1), ("negative", -1)):
        raw = validate_surface_geometry(
            asset.surface,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        reports[name] = {**compact_geometry(raw), "pass": geometry_pass(raw)}
    gates = [
        {
            "name": "full_resolution_shape_is_canonical_scale_derived_shape",
            "pass": tuple(asset.output_shape) == tuple(expected_shape_yx),
            "value": list(asset.output_shape),
            "expected": list(expected_shape_yx),
        },
        {
            "name": "full_resolution_orphan_cleanup_is_strictly_subtractive_and_quad_preserving",
            "pass": bool(
                np.all(~render_mask | initial_mask)
                and np.array_equal(active_quads(render_mask), initial_quads)
            ),
            "removed_unused_vertex_count": int(
                np.count_nonzero(initial_mask & ~render_mask)
            ),
            "removed_quad_count": 0,
        },
        {
            "name": "full_resolution_induced_quad_closure_exact",
            "pass": bool(topology["induced_quad_closure_exact"]),
        },
        {
            "name": "full_resolution_single_manifold_quad_component",
            "pass": not bool(topology["nonmanifold_risk"]),
        },
        {
            "name": "full_resolution_physical_area",
            "pass": area_cm2 >= MINIMUM_AREA_CM2,
            "value_cm2": area_cm2,
            "threshold_cm2": MINIMUM_AREA_CM2,
        },
        {"name": "full_resolution_positive_native_qc", "pass": reports["positive"]["pass"]},
        {"name": "full_resolution_negative_native_qc", "pass": reports["negative"]["pass"]},
    ]
    passed = all(bool(gate["pass"]) for gate in gates)
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "full-resolution-surface-valid-mask.tif"
    tifffile.imwrite(
        mask_path,
        asset.surface.valid.astype(np.uint8) * 255,
        photometric="minisblack",
        compression="deflate",
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": candidate_id,
        "verdict": "pass_to_public_raw_chunk_fetch" if passed else "stop_before_raw_fetch",
        "paid_compute_cost_usd": 0.0,
        "no_raw_network_before_gate": True,
        "source_stored_validation": {
            "path": str(validation_path.resolve()),
            "sha256": sha256_bytes(validation_payload),
            "verdict": validation["verdict"],
        },
        "full_resolution_tifxyz": asset.manifest(),
        "surface_interpolation": {
            "method": "canonical mask-aware linear TIFFXYZ resampling",
            "stored_shape": list(asset.stored_shape),
            "full_shape": list(asset.output_shape),
            "scale_yx": list(asset.scale_yx),
        },
        "full_resolution_strictly_subtractive_mask_cleanup": {
            "reason": (
                "canonical mask-aware resampling produced isolated valid pixels that "
                "belonged to no all-four-valid quad and therefore had no 2-D normal"
            ),
            "algorithm": "render_mask = union_of_vertices(active_quads(initial_mask))",
            "initial_mask_valid_vertex_count": int(np.count_nonzero(initial_mask)),
            "initial_mask_content_sha256": sha256_bytes(
                np.ascontiguousarray(initial_mask).view(np.uint8).tobytes()
            ),
            "removed_unused_vertex_count": int(
                np.count_nonzero(initial_mask & ~render_mask)
            ),
            "added_vertex_count": int(np.count_nonzero(render_mask & ~initial_mask)),
            "initial_quad_count": int(np.count_nonzero(initial_quads)),
            "render_quad_count": int(np.count_nonzero(active_quads(render_mask))),
            "quad_set_bit_identical": bool(
                np.array_equal(active_quads(render_mask), initial_quads)
            ),
            "initial_topology": initial_topology,
        },
        "explicit_full_resolution_valid_mask": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "array_content_sha256": sha256_bytes(
                np.ascontiguousarray(asset.surface.valid).view(np.uint8).tobytes()
            ),
            "valid_vertex_count": int(np.count_nonzero(asset.surface.valid)),
        },
        "topology": topology,
        "physical_area_cm2": area_cm2,
        "normal_offset_voxels": list(OFFSET_VOXELS),
        "normal_offset_micrometers": list(offsets_um),
        "native_geometry_by_sign": reports,
        "gate_results": gates,
        "pass_to_public_raw_chunk_fetch": passed,
        "limitations": [
            "Self-intersection remains unknown because geometric self-intersection was not computed.",
            "Combinatorial manifold topology does not prove absence of geometric self-intersection.",
        ],
    }
    _atomic_json(output_directory / "full-resolution-preflight.json", report)
    return asset, report


class CachedPublicZarrArray:
    """Minimal read-only Zarr-v2 array with disk payload cache and decoded LRU."""

    def __init__(
        self,
        *,
        root_url: str,
        array_path: str,
        spec: ZarrV2ArraySpec,
        cache_directory: Path,
        blosc_path: Path,
        decoded_lru_chunks: int = 96,
    ) -> None:
        self.root_url = root_url.rstrip("/")
        self.array_path = array_path.strip("/")
        self.spec = spec
        self.cache_directory = cache_directory
        self.blosc_path = blosc_path
        self.decoded_lru_chunks = int(decoded_lru_chunks)
        self.shape = spec.shape_zyx
        self.chunks = spec.chunks_zyx
        self.dtype = spec.dtype
        self.ndim = 3
        self.read_count = 0
        self.records: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.network_bytes = 0
        self.network_fetch_count = 0
        self.cache_hits = 0
        self._decoded: OrderedDict[tuple[int, int, int], NDArray[Any]] = OrderedDict()

    def _paths(self, index: tuple[int, int, int]) -> tuple[Path, Path]:
        base = self.cache_directory.joinpath(*(str(value) for value in index[:-1]))
        return base / f"{index[-1]}.bin", base / f"{index[-1]}.missing"

    def _fetch_public(self, url: str) -> bytes:
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": RAW_USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def _load_chunk(
        self, index: tuple[int, int, int], expected_shape: tuple[int, int, int]
    ) -> tuple[NDArray[Any], Mapping[str, Any]]:
        record = self.records.setdefault(
            index,
            {
                "chunk_index_zyx": list(index),
                "read_count": 0,
                "network_fetch_count": 0,
                "cache_hit_count": 0,
            },
        )
        record["read_count"] = int(record["read_count"]) + 1
        cached_decoded = self._decoded.get(index)
        if cached_decoded is not None:
            self._decoded.move_to_end(index)
            record["decoded_lru_hit_count"] = int(
                record.get("decoded_lru_hit_count", 0)
            ) + 1
            return cached_decoded, {"cache_level": "decoded_lru"}

        payload_path, missing_path = self._paths(index)
        url = (
            self.root_url
            + "/"
            + self.array_path
            + "/"
            + chunk_key(self.spec, index)
        )
        if missing_path.exists():
            fill = self.spec.fill_value if self.spec.fill_value is not None else 0
            array = np.full(expected_shape, fill, dtype=self.dtype)
            payload: bytes | None = None
            decoder = "zarr-v2-fill-value"
            status = "cached_missing_chunk_fill_value"
            self.cache_hits += 1
            record["cache_hit_count"] = int(record["cache_hit_count"]) + 1
        else:
            if payload_path.exists():
                payload = payload_path.read_bytes()
                status = "disk_payload_cache_hit"
                self.cache_hits += 1
                record["cache_hit_count"] = int(record["cache_hit_count"]) + 1
            else:
                try:
                    payload = self._fetch_public(url)
                except urllib.error.HTTPError as error:
                    if error.code != 404:
                        raise
                    _safe_write_bytes(missing_path, b"missing public Zarr chunk\n")
                    fill = self.spec.fill_value if self.spec.fill_value is not None else 0
                    array = np.full(expected_shape, fill, dtype=self.dtype)
                    payload = None
                    decoder = "zarr-v2-fill-value"
                    status = "network_404_fill_value"
                    self.network_fetch_count += 1
                    record["network_fetch_count"] = int(
                        record["network_fetch_count"]
                    ) + 1
                else:
                    _safe_write_bytes(payload_path, payload)
                    status = "network_downloaded"
                    self.network_bytes += len(payload)
                    self.network_fetch_count += 1
                    record["network_fetch_count"] = int(
                        record["network_fetch_count"]
                    ) + 1
            if payload is not None:
                array, decoder, decoded = decode_chunk_payload(
                    payload,
                    expected_shape,
                    self.spec,
                    blosc_path=self.blosc_path,
                )
                record.update(
                    {
                        "payload_bytes": len(payload),
                        "payload_sha256": sha256_bytes(payload),
                        "decoded_bytes": len(decoded),
                        "decoded_sha256": sha256_bytes(decoded),
                    }
                )
        record.update(
            {
                "url": url,
                "payload_cache_path": str(payload_path.resolve()),
                "missing_marker_path": str(missing_path.resolve()),
                "status_last_read": status,
                "decoder": decoder,
                "expected_shape_zyx": list(expected_shape),
            }
        )
        self._decoded[index] = array
        self._decoded.move_to_end(index)
        while len(self._decoded) > self.decoded_lru_chunks:
            self._decoded.popitem(last=False)
        return array, {"cache_level": status}

    def __getitem__(self, key: Any) -> NDArray[Any]:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("raw array requires exactly three z,y,x slices")
        lower: list[int] = []
        upper: list[int] = []
        for axis, (selection, size) in enumerate(zip(key, self.shape, strict=True)):
            if not isinstance(selection, slice) or selection.step not in (None, 1):
                raise TypeError(f"raw array axis {axis} requires a unit-step slice")
            start = 0 if selection.start is None else int(selection.start)
            stop = int(size) if selection.stop is None else int(selection.stop)
            if start < 0 or stop <= start or stop > int(size):
                raise IndexError("raw array slice lies outside the public volume")
            lower.append(start)
            upper.append(stop)
        self.read_count += 1
        roi, _records = assemble_roi(
            self.spec, lower, upper, self._load_chunk
        )
        return roi

    def manifest(self) -> dict[str, Any]:
        records = [self.records[index] for index in sorted(self.records)]
        return {
            "implementation": "bounded direct public Zarr-v2 chunk reader",
            "root_url": self.root_url,
            "array_path": self.array_path,
            "shape_zyx": list(self.shape),
            "chunks_zyx": list(self.chunks),
            "dtype": self.dtype.str,
            "cache_directory": str(self.cache_directory.resolve()),
            "decoded_lru_maximum_chunks": self.decoded_lru_chunks,
            "roi_read_count": self.read_count,
            "unique_chunk_count": len(records),
            "network_fetch_count": self.network_fetch_count,
            "network_payload_bytes": self.network_bytes,
            "disk_or_missing_cache_hit_count": self.cache_hits,
            "records": records,
        }


def validate_raw_metadata(
    root_url: str,
    array_path: str,
    *,
    expected_shape_zyx: tuple[int, int, int] = PINNED_RAW_SHAPE_ZYX,
) -> tuple[ZarrV2ArraySpec, dict[str, Any]]:
    group, group_provenance = fetch_json(root_url.rstrip("/") + "/.zgroup")
    attrs, attrs_provenance = fetch_json(root_url.rstrip("/") + "/.zattrs")
    array, array_provenance = fetch_json(
        root_url.rstrip("/") + "/" + array_path.strip("/") + "/.zarray"
    )
    if group.get("zarr_format") != 2:
        raise ValueError("official raw root is not a Zarr-v2 group")
    axes = attrs.get("multiscales", [{}])[0].get("axes", [])
    axis_names = [item.get("name") if isinstance(item, dict) else item for item in axes]
    if axis_names != ["z", "y", "x"]:
        raise ValueError(f"official raw axes changed: {axis_names!r}")
    spec = ZarrV2ArraySpec.from_metadata(array)
    if spec.shape_zyx != expected_shape_zyx:
        raise ValueError("official raw shape changed from the preflight-pinned shape")
    if spec.dtype != np.dtype(np.uint8):
        raise ValueError(f"official raw dtype changed: {spec.dtype}")
    return spec, {
        "group": group_provenance,
        "root_attributes": attrs_provenance,
        "array_metadata": array_provenance,
        "axes": axis_names,
        "shape_zyx": list(spec.shape_zyx),
        "chunks_zyx": list(spec.chunks_zyx),
        "dtype": spec.dtype.str,
        "order": spec.order,
        "dimension_separator": spec.dimension_separator,
        "compressor": spec.compressor,
        "filters": spec.filters,
    }


def window_u8(
    array: NDArray[Any], valid: NDArray[np.bool_], *, symmetric: bool = False
) -> tuple[NDArray[np.uint8], dict[str, float]]:
    values = np.asarray(array, dtype=np.float64)
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return np.zeros(values.shape, dtype=np.uint8), {"low": 0.0, "high": 1.0}
    if symmetric:
        bound = float(np.percentile(np.abs(selected), 99.0))
        bound = max(bound, 1e-12)
        low, high = -bound, bound
    else:
        low, high = (float(value) for value in np.percentile(selected, [1.0, 99.5]))
        if high <= low:
            high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~valid] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8), {"low": low, "high": high}


def save_png(path: Path, image: NDArray[np.uint8]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGB" if image.ndim == 3 else "L"
    Image.fromarray(image, mode=mode).save(path, optimize=True)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def contact_sheet(
    frames: Sequence[tuple[str, NDArray[np.uint8]]], *, columns: int = 4
) -> Image.Image:
    thumbnail = (440, 430)
    label_height = 30
    rows = math.ceil(len(frames) / columns)
    canvas = Image.new(
        "L", (columns * thumbnail[0], rows * (thumbnail[1] + label_height)), color=0
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(frames):
        image = Image.fromarray(array, mode="L")
        image.thumbnail(thumbnail, Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * thumbnail[0] + (thumbnail[0] - image.width) // 2
        y = row * (thumbnail[1] + label_height) + label_height
        canvas.paste(image, (x, y))
        draw.text((column * thumbnail[0] + 8, row * (thumbnail[1] + label_height) + 7), label, fill=255)
    return canvas


def model_window_mappings() -> dict[str, Any]:
    positive = {
        "low": list(range(0, 62)),
        "mid": list(range(15, 77)),
        "high": list(range(31, 93)),
    }
    return {
        "stored_frame_to_offset_voxels": {
            str(index): int(offset) for index, offset in enumerate(OFFSET_VOXELS)
        },
        "full_93_layer_sign_order_equivalences": {
            "positive_direct": list(range(93)),
            "positive_reversed": list(range(92, -1, -1)),
            "negative_direct": list(range(92, -1, -1)),
            "negative_reversed": list(range(93)),
        },
        "latest_62_layer_windows": {
            **{
                f"positive_{band}_direct": indices
                for band, indices in positive.items()
            },
            **{
                f"positive_{band}_reversed": list(reversed(indices))
                for band, indices in positive.items()
            },
            "negative_low_direct": list(range(92, 30, -1)),
            "negative_low_reversed": list(range(31, 93)),
            "negative_mid_direct": list(range(77, 15, -1)),
            "negative_mid_reversed": list(range(16, 78)),
            "negative_high_direct": list(range(61, -1, -1)),
            "negative_high_reversed": list(range(62)),
        },
        "even_window_note": (
            "The 62-layer mid band is not self-symmetric: negative mid uses stored "
            "frames 77..16 (offsets +31..-30), while positive mid uses 15..76 "
            "(offsets -31..+30). Both are exact slices of the 93-layer stack."
        ),
    }


def validate_cache_seed_manifest(
    seed_manifest_path: Path, raw_cache: Path
) -> dict[str, Any]:
    """Fail closed unless the raw cache exactly matches its pinned seed set."""

    payload = seed_manifest_path.read_bytes()
    seed = json.loads(payload)
    if seed.get("status") != "complete":
        raise ValueError("raw cache seed manifest is not complete")
    if Path(seed["destination_cache"]).resolve() != raw_cache.resolve():
        raise ValueError("raw cache path differs from seed manifest destination")
    records = seed["records"]
    if int(seed["verified_payload_count"]) != len(records):
        raise ValueError("raw cache seed count disagrees with its records")
    expected_paths: set[Path] = set()
    total_bytes = 0
    for record in records:
        index = tuple(int(value) for value in record["chunk_index_zyx"])
        if len(index) != 3:
            raise ValueError("raw cache seed contains an invalid chunk index")
        path = raw_cache / str(index[0]) / str(index[1]) / f"{index[2]}.bin"
        expected_paths.add(path.resolve())
        expected_bytes = int(record["bytes"])
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"raw cache seed payload missing/size mismatch: {path}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"raw cache seed payload hash mismatch: {path}")
        total_bytes += expected_bytes
    actual_paths = {path.resolve() for path in raw_cache.rglob("*.bin")}
    if actual_paths != expected_paths:
        raise ValueError(
            "raw cache contains payloads outside the exact verified seed set"
        )
    if any(raw_cache.rglob("*.missing")):
        raise ValueError("raw cache seed contains missing-chunk markers")
    if total_bytes != int(seed["verified_payload_bytes"]):
        raise ValueError("raw cache seed byte total disagrees with its records")
    return {
        "path": str(seed_manifest_path.resolve()),
        "sha256": sha256_bytes(payload),
        "verified_payload_count": len(records),
        "verified_payload_bytes": total_bytes,
        "exact_cache_set_match_before_raw_access": True,
    }


def build_diagnostics(stack_directory: Path, output_directory: Path) -> dict[str, Any]:
    layer_paths = [stack_directory / "positive" / f"{index:02d}.tif" for index in range(93)]
    if not all(path.is_file() for path in layer_paths):
        raise FileNotFoundError("raw renderer did not produce all 93 positive layers")
    valid = tifffile.imread(stack_directory / "positive/valid-all.tif") != 0
    first = tifffile.memmap(layer_paths[0])
    total = np.zeros(first.shape, dtype=np.float64)
    total_square = np.zeros(first.shape, dtype=np.float64)
    minimum = np.full(first.shape, 255, dtype=np.uint8)
    maximum = np.zeros(first.shape, dtype=np.uint8)
    far_total = np.zeros(first.shape, dtype=np.float64)
    far_count = 0
    selected_contact: dict[int, NDArray[np.uint8]] = {}
    contact_indices = {0, 15, 31, 46, 61, 77, 92}
    rgb_indices = (15, 46, 77)
    rgb_sources: dict[int, NDArray[np.uint8]] = {}
    for index, path in enumerate(layer_paths):
        layer = np.asarray(tifffile.memmap(path), dtype=np.uint8)
        numeric = layer.astype(np.float64)
        total += numeric
        total_square += numeric * numeric
        np.minimum(minimum, layer, out=minimum)
        np.maximum(maximum, layer, out=maximum)
        if index < 16 or index >= 77:
            far_total += numeric
            far_count += 1
        if index in contact_indices:
            selected_contact[index] = layer.copy()
        if index in rgb_indices:
            rgb_sources[index] = layer.copy()
    mean = total / 93.0
    variance = np.maximum(total_square / 93.0 - mean * mean, 0.0)
    standard_deviation = np.sqrt(variance)
    dynamic_range = maximum.astype(np.float64) - minimum.astype(np.float64)
    center = selected_contact[46].astype(np.float64)
    far_mean = far_total / float(far_count)
    center_minus_far = center - far_mean
    local_background = ndimage.gaussian_filter(center, sigma=8.0, mode="nearest")
    local_residual = center - local_background

    diagnostic_directory = output_directory / "diagnostics"
    records: dict[str, Any] = {}
    transformations: dict[str, Any] = {}
    derived = {
        "center_offset0": (center, False),
        "depth_mean": (mean, False),
        "depth_standard_deviation": (standard_deviation, False),
        "depth_range": (dynamic_range, False),
        "center_minus_far_depth_mean": (center_minus_far, True),
        "center_local_residual": (local_residual, True),
        "center_local_dark_residual": (-local_residual, False),
        "center_local_bright_residual": (local_residual, False),
    }
    for name, (array, symmetric) in derived.items():
        image, window = window_u8(array, valid, symmetric=symmetric)
        records[name] = save_png(diagnostic_directory / f"{name}.png", image)
        transformations[name] = {
            "display_only_window": window,
            "symmetric_diverging_window": symmetric,
        }
    contact_frames: list[tuple[str, NDArray[np.uint8]]] = []
    for index in sorted(selected_contact):
        image, _window = window_u8(selected_contact[index], valid)
        contact_frames.append((f"frame {index:02d} / offset {OFFSET_VOXELS[index]:+d} vox", image))
    sheet = contact_sheet(contact_frames)
    contact_path = diagnostic_directory / "selected-offset-contact-sheet.png"
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_path, optimize=True)
    records["selected_offset_contact_sheet"] = {
        "path": str(contact_path.resolve()),
        "sha256": sha256_file(contact_path),
        "bytes": contact_path.stat().st_size,
        "frames": sorted(selected_contact),
        "offsets_voxels": [OFFSET_VOXELS[index] for index in sorted(selected_contact)],
    }
    rgb_channels = []
    rgb_windows = []
    for index in rgb_indices:
        channel, window = window_u8(rgb_sources[index], valid)
        rgb_channels.append(channel)
        rgb_windows.append(window)
    rgb = np.stack(rgb_channels, axis=-1)
    records["fiber_depth_rgb_minus31_0_plus31"] = save_png(
        diagnostic_directory / "fiber-depth-rgb-minus31-0-plus31.png", rgb
    )
    transformations["fiber_depth_rgb_minus31_0_plus31"] = {
        "channels_frame_indices": list(rgb_indices),
        "channels_offset_voxels": [OFFSET_VOXELS[index] for index in rgb_indices],
        "per_channel_display_windows": rgb_windows,
    }
    return {
        "role": "deterministic visibility aids only; no ink inference or semantic claim",
        "valid_mask_vertex_count": int(np.count_nonzero(valid)),
        "records": records,
        "transformations": transformations,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    preflight_path = output / "full-resolution-preflight.json"
    if output.exists() and (output / "raw-stack-manifest.json").exists() and not args.overwrite:
        raise FileExistsError("completed raw output already exists; pass --overwrite to replace it")
    asset, preflight = full_resolution_preflight(
        Path(args.surface),
        Path(args.validation),
        output,
        candidate_id=str(args.candidate_id),
        expected_shape_yx=tuple(int(value) for value in args.expected_shape_yx),
        resample_shape_yx=(
            tuple(int(value) for value in args.resample_shape_yx)
            if args.resample_shape_yx is not None
            else None
        ),
        voxel_um=float(args.voxel_um),
        volume_shape_zyx=tuple(int(value) for value in args.volume_shape_zyx),
        accepted_validation_verdicts=(str(args.accepted_validation_verdict),),
    )
    if not preflight["pass_to_public_raw_chunk_fetch"]:
        return {
            "schema_version": 1,
            "status": "stopped_before_raw_fetch",
            "verdict": "full_resolution_preflight_failed",
            "full_resolution_preflight": str(preflight_path.resolve()),
            "paid_compute_cost_usd": 0.0,
        }
    if args.preflight_only:
        return {
            "schema_version": 1,
            "status": "preflight_complete",
            "verdict": "pass_to_public_raw_chunk_fetch",
            "full_resolution_preflight": str(preflight_path.resolve()),
            "paid_compute_cost_usd": 0.0,
        }

    self_intersection_record = None
    if args.self_intersection_audit is not None:
        audit_path = Path(args.self_intersection_audit)
        audit_payload = audit_path.read_bytes()
        audit = json.loads(audit_payload)
        if not bool(audit.get("pass_to_raw_diagnostic")):
            raise ValueError("self-intersection audit does not authorize raw rendering")
        if audit.get("surface", {}).get("input_sha256") != asset.input_sha256:
            raise ValueError("self-intersection audit is not bound to this TIFFXYZ input")
        self_intersection_record = {
            "path": str(audit_path.resolve()),
            "sha256": sha256_bytes(audit_payload),
            "stored_nonadjacent_intersection_count": int(
                audit["nonadjacent_intersection_count"]
            ),
            "full_res_nonadjacent_clearance_lower_bound_voxels": float(
                audit["certified_full_res_nonadjacent_clearance_lower_bound_voxels"]
            ),
            "full_res_nonadjacent_clearance_lower_bound_um": float(
                audit["certified_full_res_nonadjacent_clearance_lower_bound_um"]
            ),
            "scope": "non-adjacent triangles; shared-source-vertex pairs excluded",
        }

    cache_seed = (
        validate_cache_seed_manifest(
            Path(args.cache_seed_manifest), Path(args.raw_cache)
        )
        if args.cache_seed_manifest is not None
        else None
    )
    spec, metadata = validate_raw_metadata(
        str(args.raw_root),
        str(args.array_path),
        expected_shape_zyx=tuple(int(value) for value in args.volume_shape_zyx),
    )
    blosc_path = Path(args.blosc)
    if not blosc_path.is_file():
        raise FileNotFoundError(f"pinned Blosc codec not found: {blosc_path}")
    volume = CachedPublicZarrArray(
        root_url=str(args.raw_root),
        array_path=str(args.array_path),
        spec=spec,
        cache_directory=Path(args.raw_cache),
        blosc_path=blosc_path,
        decoded_lru_chunks=int(args.decoded_lru_chunks),
    )
    stack_directory = output / "raw-surface-stack"
    options = RenderOptions(
        voxel_size_zyx_um=(float(args.voxel_um),) * 3,
        offsets_um=tuple(float(value) * float(args.voxel_um) for value in OFFSET_VOXELS),
        signs=("positive",),
        tile_shape=(int(args.tile_size), int(args.tile_size)),
        output_dtype="uint8",
        fill_value=0.0,
        png_mode="preview",
        overwrite=bool(args.overwrite),
        hash_outputs=True,
    )

    def progress(index: int, total: int, detail: Mapping[str, Any]) -> None:
        print(json.dumps({"tile": index, "tiles": total, **detail}), flush=True)

    render_manifest = render_surface_to_directory(
        volume,
        asset.surface,
        stack_directory,
        options=options,
        tifxyz_manifest=asset.manifest(),
        volume_manifest={
            "source": str(args.raw_root),
            "array_path": str(args.array_path),
            **metadata,
        },
        progress=progress,
    )
    full_mask = tifffile.imread(output / "full-resolution-surface-valid-mask.tif") != 0
    valid_all_path = stack_directory / "positive/valid-all.tif"
    render_valid_all = tifffile.imread(valid_all_path) != 0
    mask_identity = bool(np.array_equal(full_mask, render_valid_all))
    if not mask_identity:
        raise RuntimeError("renderer valid-all mask differs from the precommitted full mask")
    diagnostics = build_diagnostics(stack_directory, output)
    mappings = model_window_mappings()
    final = {
        "schema_version": 1,
        "status": "complete",
        "verdict": "model_ready_free_raw_stack",
        "candidate_id": str(args.candidate_id),
        "paid_compute_cost_usd": 0.0,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
            "invocation": {
                "candidate_id": str(args.candidate_id),
                "surface": str(Path(args.surface).resolve()),
                "validation": str(Path(args.validation).resolve()),
                "output": str(output.resolve()),
                "raw_cache": str(Path(args.raw_cache).resolve()),
                "cache_seed_manifest": (
                    str(Path(args.cache_seed_manifest).resolve())
                    if args.cache_seed_manifest is not None
                    else None
                ),
                "expected_shape_yx": [
                    int(value) for value in args.expected_shape_yx
                ],
                "resample_shape_yx": (
                    [int(value) for value in args.resample_shape_yx]
                    if args.resample_shape_yx is not None
                    else None
                ),
                "voxel_um": float(args.voxel_um),
                "volume_shape_zyx": [int(value) for value in args.volume_shape_zyx],
                "self_intersection_audit": (
                    str(Path(args.self_intersection_audit).resolve())
                    if args.self_intersection_audit is not None
                    else None
                ),
                "tile_size": int(args.tile_size),
                "decoded_lru_chunks": int(args.decoded_lru_chunks),
            },
        },
        "full_resolution_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
            "verdict": preflight["verdict"],
            "physical_area_cm2": preflight["physical_area_cm2"],
        },
        "self_intersection_audit": self_intersection_record,
        "raw_public_zarr": {
            "metadata": metadata,
            "cache_seed": cache_seed,
            "chunk_reader": volume.manifest(),
            "codec_path": str(blosc_path.resolve()),
            "codec_sha256": sha256_file(blosc_path),
        },
        "raw_stack": {
            "directory": str((stack_directory / "positive").resolve()),
            "renderer_manifest_path": str((stack_directory / "manifest.json").resolve()),
            "renderer_manifest_sha256": sha256_file(stack_directory / "manifest.json"),
            "shape_frame_row_column": [93, *asset.output_shape],
            "dtype": "uint8",
            "lossless_raw_tiff": True,
            "physical_positive_normal_definition": (
                "cross(dP/dcolumn, dP/drow) in xyz physical coordinates"
            ),
            "stored_offsets_voxels": list(OFFSET_VOXELS),
            "stored_offsets_micrometers": [
                float(value) * float(args.voxel_um) for value in OFFSET_VOXELS
            ],
            "interpolation": "trilinear raw volume sampling",
            "surface_resolution": "canonical full linear TIFFXYZ raster",
            "explicit_surface_valid_mask": preflight["explicit_full_resolution_valid_mask"],
            "renderer_valid_all_mask": {
                "path": str(valid_all_path.resolve()),
                "sha256": sha256_file(valid_all_path),
                "array_identical_to_explicit_surface_mask": mask_identity,
            },
            "renderer": render_manifest,
        },
        "sign_order_and_model_window_mappings": mappings,
        "diagnostics": diagnostics,
        "limitations": [
            "No detector, ink model, OCR, or semantic text inference was run.",
            "Direct-ink diagnostics are display-only raw contrasts, not evidence of letters.",
            (
                "Self-intersection scope is bound to the recorded non-adjacent stored-triangle "
                "audit and conservative full-raster clearance bound; local shared-simplex "
                "behavior is covered by the independent full-resolution fold/topology gates."
                if self_intersection_record is not None
                else "Self-intersection remains unknown_not_computed."
            ),
            "Public raw access is free; recorded paid compute cost is zero.",
        ],
    }
    _atomic_json(output / "raw-stack-manifest.json", final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument(
        "--expected-shape-yx", type=int, nargs=2, default=(1860, 1900)
    )
    parser.add_argument("--resample-shape-yx", type=int, nargs=2)
    parser.add_argument("--voxel-um", type=float, default=VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx", type=int, nargs=3, default=PINNED_RAW_SHAPE_ZYX
    )
    parser.add_argument(
        "--accepted-validation-verdict", default="pass_to_free_raw_render"
    )
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--array-path", default=DEFAULT_ARRAY_PATH)
    parser.add_argument("--raw-cache", type=Path, default=DEFAULT_OUTPUT / "raw-chunk-cache")
    parser.add_argument("--cache-seed-manifest", type=Path)
    parser.add_argument("--self-intersection-audit", type=Path)
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--decoded-lru-chunks", type=int, default=96)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "error",
            "candidate_id": str(args.candidate_id),
            "verdict": "render_error",
            "paid_compute_cost_usd": 0.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(output / "raw-stack-failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "output": str(output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
