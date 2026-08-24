#!/usr/bin/env python3
"""Dependency-light public-m7 validator for a small TIFFXYZ patch.

This validator intentionally does not import the Python ``zarr`` package.  It
reads the official Zarr-v2 metadata, derives the exact source chunks touched by
the requested trilinear samples, downloads each chunk at most once, and decodes
Blosc with VC3D's bundled library.  Surface normals and quantization match the
existing First Letters native preflight.

The verdict is only an intermediate geometry-growth decision.  It never
authorizes ink inference and cannot replace final full-resolution surface and
self-intersection validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from native_surface_sampler import sample_trilinear_zyx, validate_surface_geometry
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOLUME_SHAPE_ZYX,
    DEFAULT_VOXEL_UM,
    derive_geometry_masks,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import (
    ZarrV2ArraySpec,
    chunk_shape,
    decode_chunk_payload,
    fetch_bytes,
    fetch_json,
    sha256_bytes,
    sha256_file,
)
from tifxyz_render_pipeline import load_tifxyz_asset, sanitize_source


DEFAULT_M7_ROOT = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/"
    "representations/predictions/surfaces/"
    "20250521151220-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
DEFAULT_PATCH = Path(
    "outputs/first-letters-geometry/vc3d-interactive/"
    "priority1-seed-3526-4188-14072/growpatch-5gen/"
    "auto_grown_20260809025149671"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/first-letters-geometry/vc3d-interactive/"
    "priority1-seed-3526-4188-14072/growpatch-5gen-public-m7"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


ChunkFetcher = Callable[[str], bytes]


class PublicZarrChunkSampler:
    """Trilinear sampler backed by exact public Zarr-v2 chunk requests."""

    def __init__(
        self,
        *,
        root_url: str,
        array_path: str,
        spec: ZarrV2ArraySpec,
        blosc_path: Path = DEFAULT_BLOSC,
        fetcher: ChunkFetcher = fetch_bytes,
    ) -> None:
        self.root_url = root_url.rstrip("/")
        self.array_path = array_path.strip("/")
        self.spec = spec
        self.shape_zyx = spec.shape_zyx
        self.chunks_zyx = spec.chunks_zyx
        self.blosc_path = blosc_path
        self.fetcher = fetcher
        self._cache: dict[tuple[int, int, int], NDArray[Any]] = {}
        self._records: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.sample_call_count = 0
        self.requested_valid_point_count = 0

    def _chunk_url(self, index: Sequence[int]) -> str:
        separator = self.spec.dimension_separator
        key = separator.join(str(int(value)) for value in index)
        return f"{self.root_url}/{self.array_path}/{key}"

    def _download_one(
        self, index: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], NDArray[Any], dict[str, Any]]:
        expected_shape = chunk_shape(self.spec, index)
        url = self._chunk_url(index)
        try:
            payload = self.fetcher(url)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            fill = self.spec.fill_value if self.spec.fill_value is not None else 0
            array = np.full(expected_shape, fill, dtype=self.spec.dtype)
            decoded = array.tobytes(order=self.spec.order)
            record = {
                "chunk_index_zyx": list(index),
                "chunk_shape_zyx": list(expected_shape),
                "url": url,
                "status": "missing_chunk_fill_value",
                "downloaded_bytes": 0,
                "payload_sha256": None,
                "decoded_bytes": len(decoded),
                "decoded_sha256": sha256_bytes(decoded),
                "decoder": "zarr-v2-fill-value",
            }
            return index, array, record
        array, decoder, decoded = decode_chunk_payload(
            payload,
            expected_shape,
            self.spec,
            blosc_path=self.blosc_path,
        )
        record = {
            "chunk_index_zyx": list(index),
            "chunk_shape_zyx": list(expected_shape),
            "url": url,
            "status": "downloaded",
            "downloaded_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "decoded_bytes": len(decoded),
            "decoded_sha256": sha256_bytes(decoded),
            "decoder": decoder,
        }
        return index, array, record

    def _ensure_chunks(self, indices: NDArray[np.int64]) -> None:
        requested = [tuple(int(value) for value in row) for row in indices]
        missing = [index for index in requested if index not in self._cache]
        if not missing:
            return
        workers = min(8, len(missing))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(self._download_one, missing))
        for index, array, record in loaded:
            self._cache[index] = array
            self._records[index] = record

    def sample(
        self,
        coordinates_xyz: NDArray[np.float64],
        *,
        point_valid: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        coordinates = np.asarray(coordinates_xyz, dtype=np.float64)
        requested = np.asarray(point_valid, dtype=bool)
        if coordinates.shape[:-1] != requested.shape or coordinates.shape[-1] != 3:
            raise ValueError("coordinate and validity shapes do not match")
        self.sample_call_count += 1
        point_shape = requested.shape
        flat = coordinates.reshape(-1, 3)
        flat_requested = requested.reshape(-1)
        z_size, y_size, x_size = self.shape_zyx
        finite = np.isfinite(flat).all(axis=1)
        in_bounds = (
            finite
            & (flat[:, 0] >= 0.0)
            & (flat[:, 0] <= x_size - 1)
            & (flat[:, 1] >= 0.0)
            & (flat[:, 1] <= y_size - 1)
            & (flat[:, 2] >= 0.0)
            & (flat[:, 2] <= z_size - 1)
        )
        usable = flat_requested & in_bounds
        output = np.zeros(flat.shape[0], dtype=np.float32)
        output_valid = np.zeros(flat.shape[0], dtype=bool)
        selected_indices = np.flatnonzero(usable)
        self.requested_valid_point_count += int(selected_indices.size)
        if not selected_indices.size:
            return output.reshape(point_shape), output_valid.reshape(point_shape)

        selected_xyz = flat[selected_indices]
        lower_xyz = np.floor(selected_xyz).astype(np.int64)
        upper_xyz = np.minimum(
            lower_xyz + 1,
            np.asarray((x_size - 1, y_size - 1, z_size - 1), dtype=np.int64),
        )
        fractions = selected_xyz - lower_xyz
        point_rows: list[NDArray[np.int64]] = []
        voxel_rows: list[NDArray[np.int64]] = []
        weight_rows: list[NDArray[np.float64]] = []
        point_index = np.arange(selected_indices.size, dtype=np.int64)
        for z_side in (0, 1):
            for y_side in (0, 1):
                for x_side in (0, 1):
                    sides = np.asarray((x_side, y_side, z_side), dtype=np.int64)
                    corner_xyz = np.where(
                        sides[None, :] == 0, lower_xyz, upper_xyz
                    )
                    component_weight = np.where(
                        sides[None, :] == 0, 1.0 - fractions, fractions
                    )
                    weights = np.prod(component_weight, axis=1)
                    contributes = weights > 0.0
                    point_rows.append(point_index[contributes])
                    voxel_rows.append(corner_xyz[contributes, ::-1])
                    weight_rows.append(weights[contributes])
        corner_points = np.concatenate(point_rows)
        corner_zyx = np.concatenate(voxel_rows)
        corner_weights = np.concatenate(weight_rows)
        chunks = np.asarray(self.chunks_zyx, dtype=np.int64)
        chunk_ids = corner_zyx // chunks
        unique, inverse = np.unique(chunk_ids, axis=0, return_inverse=True)
        self._ensure_chunks(unique)
        accumulated = np.zeros(selected_indices.size, dtype=np.float64)
        for group_index, chunk_id in enumerate(unique):
            index = tuple(int(value) for value in chunk_id)
            chunk = self._cache[index]
            chunk_start = chunk_id * chunks
            entries = np.flatnonzero(inverse == group_index)
            local = corner_zyx[entries] - chunk_start
            values = np.asarray(
                chunk[local[:, 0], local[:, 1], local[:, 2]], dtype=np.float64
            )
            np.add.at(
                accumulated,
                corner_points[entries],
                corner_weights[entries] * values,
            )
        output[selected_indices] = accumulated.astype(np.float32)
        output_valid[selected_indices] = True
        return output.reshape(point_shape), output_valid.reshape(point_shape)

    def manifest(self) -> dict[str, Any]:
        records = [self._records[index] for index in sorted(self._records)]
        aggregate_lines = [
            f"{record['chunk_index_zyx']}:{record['decoded_sha256']}"
            for record in records
        ]
        aggregate = hashlib.sha256(
            ("\n".join(aggregate_lines) + "\n").encode("utf-8")
        ).hexdigest()
        return {
            "implementation": "direct Zarr-v2 exact-chunk trilinear sampler; no Python zarr",
            "shape_zyx": list(self.shape_zyx),
            "chunks_zyx": list(self.chunks_zyx),
            "sample_call_count": self.sample_call_count,
            "requested_valid_point_count": self.requested_valid_point_count,
            "unique_chunk_count": len(records),
            "downloaded_bytes": int(
                sum(int(record["downloaded_bytes"]) for record in records)
            ),
            "each_chunk_fetched_at_most_once": True,
            "decompressed_chunk_aggregate_sha256": aggregate,
            "chunks": records,
        }


def contiguous_runs_inclusive_offsets(
    trace: NDArray[np.bool_], half_length: int
) -> list[list[int]]:
    padded = np.pad(np.asarray(trace, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        [int(start - half_length), int(stop - 1 - half_length)]
        for start, stop in zip(starts, stops)
    ]


def classify_transect(trace: NDArray[np.bool_]) -> tuple[str, list[list[int]]]:
    center = trace.size // 2
    runs = contiguous_runs_inclusive_offsets(trace, center)
    if not runs:
        return "zero_run", runs
    if len(runs) > 1:
        return "multi_run", runs
    if not bool(trace[center]):
        return "off_center", runs
    return "single_run_centered", runs


def evaluate_valid_vertex_transects(
    frames: NDArray[np.uint8],
    vertex_valid: NDArray[np.bool_],
    *,
    seed: int,
    sample_count: int = 50,
) -> dict[str, Any]:
    valid = np.asarray(vertex_valid, dtype=bool)
    if frames.ndim != 3 or frames.shape[1:] != valid.shape:
        raise ValueError("frames and vertex_valid shapes do not match")
    candidates = np.flatnonzero(valid)
    if candidates.size < sample_count:
        return {
            "evaluated": False,
            "reason": f"only {candidates.size} valid vertices, fewer than {sample_count}",
            "eligible_vertex_count": int(candidates.size),
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(candidates, size=sample_count, replace=False))
    categories = {
        "single_run_centered": 0,
        "zero_run": 0,
        "multi_run": 0,
        "off_center": 0,
    }
    records: list[dict[str, Any]] = []
    width = valid.shape[1]
    foreground = frames > 0
    for flat_index in selected:
        row, column = divmod(int(flat_index), width)
        category, runs = classify_transect(foreground[:, row, column])
        categories[category] += 1
        records.append(
            {
                "flat_index": int(flat_index),
                "row": row,
                "column": column,
                "category": category,
                "runs_inclusive_offsets": runs,
            }
        )
    return {
        "evaluated": True,
        "rng": "numpy.default_rng",
        "rng_seed": int(seed),
        "eligible_vertex_count": int(candidates.size),
        "sample_count": int(sample_count),
        "chosen_flat_indices": selected.astype(int).tolist(),
        "category_counts": categories,
        "pass_count": categories["single_run_centered"],
        "pass": categories["single_run_centered"] == sample_count,
        "records": records,
    }


def evaluate_all_vertex_transects(
    frames: NDArray[np.uint8], vertex_valid: NDArray[np.bool_]
) -> dict[str, Any]:
    valid = np.asarray(vertex_valid, dtype=bool)
    foreground = frames > 0
    categories = {
        "single_run_centered": 0,
        "zero_run": 0,
        "multi_run": 0,
        "off_center": 0,
    }
    failures: list[dict[str, Any]] = []
    for row, column in np.argwhere(valid):
        category, runs = classify_transect(foreground[:, row, column])
        categories[category] += 1
        if category != "single_run_centered":
            failures.append(
                {
                    "row": int(row),
                    "column": int(column),
                    "category": category,
                    "runs_inclusive_offsets": runs,
                }
            )
    total = int(np.count_nonzero(valid))
    return {
        "eligible_vertex_count": total,
        "category_counts": categories,
        "clean_fraction": (
            float(categories["single_run_centered"] / total) if total else 0.0
        ),
        "all_clean": categories["single_run_centered"] == total and total > 0,
        "failure_records": failures,
        "gate_role": "diagnostic only; canonical gate is deterministic 50/50",
    }


def support_summary(
    frames: NDArray[np.uint8],
    offsets: Sequence[int],
    vertex_valid: NDArray[np.bool_],
) -> dict[str, Any]:
    valid = np.asarray(vertex_valid, dtype=bool)
    offset_array = np.asarray(offsets, dtype=np.int64)
    if frames.shape[0] != offset_array.size or frames.shape[1:] != valid.shape:
        raise ValueError("support frame, offset, and vertex shapes differ")
    foreground = frames > 0
    supported = np.any(foreground, axis=0) & valid
    center_matches = np.flatnonzero(offset_array == 0)
    if center_matches.size != 1:
        raise ValueError("support offsets must contain zero exactly once")
    centered = foreground[int(center_matches[0])] & valid
    valid_count = int(np.count_nonzero(valid))
    supported_count = int(np.count_nonzero(supported))
    centered_count = int(np.count_nonzero(centered))
    return {
        "offsets_voxels": offset_array.astype(int).tolist(),
        "valid_vertex_count": valid_count,
        "supported_vertex_count": supported_count,
        "support_fraction": float(supported_count / valid_count) if valid_count else 0.0,
        "offset0_supported_vertex_count": centered_count,
        "offset0_support_fraction": (
            float(centered_count / valid_count) if valid_count else 0.0
        ),
        "per_offset_supported_vertex_count": {
            str(int(offset)): int(np.count_nonzero(frame & valid))
            for offset, frame in zip(offset_array, foreground)
        },
        "unsupported_row_column": [
            [int(row), int(column)] for row, column in np.argwhere(valid & ~supported)
        ],
    }


def generation_support_summary(
    generations: NDArray[Any],
    valid: NDArray[np.bool_],
    support_frames: NDArray[np.uint8],
    support_offsets: Sequence[int],
    all_frames: NDArray[np.uint8],
) -> dict[str, Any]:
    if generations.shape != valid.shape:
        raise ValueError("generations.tif shape differs from surface")
    values = np.asarray(generations)
    valid_generations = sorted(int(value) for value in np.unique(values[valid]))
    groups: dict[str, Any] = {}
    for generation in valid_generations:
        mask = valid & (values == generation)
        groups[str(generation)] = {
            "vertex_count": int(np.count_nonzero(mask)),
            "support": support_summary(support_frames, support_offsets, mask),
            "transects": evaluate_all_vertex_transects(all_frames, mask),
        }
    minimum = min(valid_generations) if valid_generations else None
    seed_mask = valid & (values == minimum) if minimum is not None else np.zeros_like(valid)
    return {
        "valid_generation_values": valid_generations,
        "minimum_valid_generation_interpreted_as_seed_core": minimum,
        "generation_zero_valid_vertex_count": int(np.count_nonzero(valid & (values == 0))),
        "seed_core": {
            "vertex_count": int(np.count_nonzero(seed_mask)),
            "support": support_summary(support_frames, support_offsets, seed_mask),
            "transects": evaluate_all_vertex_transects(all_frames, seed_mask),
        },
        "by_generation": groups,
    }


def closest_point_on_triangle(
    point_xyz: Sequence[float],
    a_xyz: Sequence[float],
    b_xyz: Sequence[float],
    c_xyz: Sequence[float],
) -> NDArray[np.float64]:
    """Return the closest point on a triangle (Ericson region tests)."""

    point = np.asarray(point_xyz, dtype=np.float64)
    a = np.asarray(a_xyz, dtype=np.float64)
    b = np.asarray(b_xyz, dtype=np.float64)
    c = np.asarray(c_xyz, dtype=np.float64)
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0 and d2 <= 0:
        return a.copy()
    bp = point - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0 and d4 <= d3:
        return b.copy()
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        weight = d1 / (d1 - d3)
        return a + weight * ab
    cp = point - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0 and d5 <= d6:
        return c.copy()
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        weight = d2 / (d2 - d6)
        return a + weight * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        weight = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + weight * (c - b)
    denominator = va + vb + vc
    if abs(denominator) <= 1e-18:
        # Degenerate input is not expected after the geometry gate, but a
        # deterministic vertex fallback keeps this diagnostic total.
        vertices = np.stack((a, b, c))
        return vertices[int(np.argmin(np.linalg.norm(vertices - point, axis=1)))].copy()
    inverse = 1.0 / denominator
    v = vb * inverse
    w = vc * inverse
    return a + v * ab + w * ac


def seed_surface_alignment(
    surface: Any, seed_xyz: Sequence[float], *, voxel_um: float
) -> dict[str, Any]:
    """Measure metadata-seed distance to the stored triangulated patch."""

    seed = np.asarray(seed_xyz, dtype=np.float64)
    if seed.shape != (3,) or not np.isfinite(seed).all():
        raise ValueError("metadata seed must be a finite xyz triple")
    points = surface.points_xyz
    valid = surface.valid
    best: tuple[float, int, int, int, NDArray[np.float64]] | None = None
    triangle_count = 0
    for row in range(surface.shape[0] - 1):
        for column in range(surface.shape[1] - 1):
            if not bool(valid[row : row + 2, column : column + 2].all()):
                continue
            triangles = (
                (points[row, column], points[row, column + 1], points[row + 1, column + 1]),
                (points[row, column], points[row + 1, column + 1], points[row + 1, column]),
            )
            for triangle_index, triangle in enumerate(triangles):
                triangle_count += 1
                closest = closest_point_on_triangle(seed, *triangle)
                distance = float(np.linalg.norm(seed - closest))
                candidate = (distance, row, column, triangle_index, closest)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    if best is None:
        return {
            "evaluated": False,
            "reason": "no fully valid stored-grid triangles",
            "valid_triangle_count": 0,
        }
    distance, row, column, triangle_index, closest = best
    return {
        "evaluated": True,
        "metadata_seed_xyz": seed.tolist(),
        "valid_triangle_count": triangle_count,
        "nearest_triangle": {
            "quad_row": row,
            "quad_column": column,
            "triangle_index": triangle_index,
        },
        "nearest_surface_point_xyz": closest.tolist(),
        "seed_minus_surface_xyz_voxels": (seed - closest).tolist(),
        "distance_voxels": distance,
        "distance_um": float(distance * voxel_um),
        "gate_role": "diagnostic; m7 support and transects determine the growth verdict",
    }


def decide_intermediate_gate(
    *,
    hard_geometry_pass: bool,
    sampling_complete: bool,
    support_fraction: float,
    minimum_support: float,
    seed_support_fraction: float,
    minimum_seed_support: float,
    transects_pass: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    gates = [
        {"name": "hard_geometry", "pass": bool(hard_geometry_pass)},
        {"name": "all_requested_m7_samples_valid", "pass": bool(sampling_complete)},
        {
            "name": "five_offset_m7_support",
            "threshold": float(minimum_support),
            "value": float(support_fraction),
            "pass": support_fraction >= minimum_support,
        },
        {
            "name": "seed_core_five_offset_m7_support",
            "threshold": float(minimum_seed_support),
            "value": float(seed_support_fraction),
            "pass": seed_support_fraction >= minimum_seed_support,
        },
        {"name": "deterministic_50_of_50_clean_transects", "pass": bool(transects_pass)},
    ]
    return all(bool(gate["pass"]) for gate in gates), gates


def validate_metadata_axes(attributes: Mapping[str, Any], array_path: str) -> None:
    multiscales = attributes.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("m7 root has no multiscales metadata")
    axes = multiscales[0].get("axes")
    names = [axis.get("name") for axis in axes] if isinstance(axes, list) else []
    if names != ["z", "y", "x"]:
        raise ValueError(f"m7 axes are {names!r}, expected ['z', 'y', 'x']")
    datasets = multiscales[0].get("datasets")
    paths = [str(item.get("path")) for item in datasets] if isinstance(datasets, list) else []
    if array_path not in paths:
        raise ValueError(f"m7 array path {array_path!r} is absent from multiscales")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    patch = Path(args.tifxyz)
    asset = load_tifxyz_asset(patch, resolution="stored")
    root = str(args.m7_root).rstrip("/")
    array_path = str(args.m7_array_path).strip("/")
    group, group_provenance = fetch_json(root + "/.zgroup")
    array_metadata, array_provenance = fetch_json(
        root + "/" + array_path + "/.zarray"
    )
    attributes, attributes_provenance = fetch_json(root + "/.zattrs")
    if group.get("zarr_format") != 2:
        raise ValueError("m7 root is not a Zarr-v2 group")
    validate_metadata_axes(attributes, array_path)
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)
    if spec.shape_zyx != tuple(int(value) for value in args.volume_shape_zyx):
        raise ValueError(
            f"m7 shape {spec.shape_zyx} differs from expected {args.volume_shape_zyx}"
        )
    if spec.chunks_zyx != (192, 192, 192):
        raise ValueError(f"m7 chunks are {spec.chunks_zyx}, expected (192, 192, 192)")
    if spec.dtype != np.dtype(np.uint8):
        raise ValueError(f"m7 dtype is {spec.dtype}, expected uint8")
    if spec.compressor is None or spec.compressor.get("id") != "blosc":
        raise ValueError("m7 array is not Blosc-compressed")

    masks = derive_geometry_masks(
        asset.surface,
        volume_shape_zyx=spec.shape_zyx,
        voxel_um=float(args.voxel_um),
        maximum_offset_voxels=15,
    )
    offsets_um = [offset * float(args.voxel_um) for offset in DEFAULT_TRANSECT_OFFSETS]
    native_reports = {
        label: validate_surface_geometry(
            asset.surface,
            volume_shape_zyx=spec.shape_zyx,
            voxel_size_zyx_um=(float(args.voxel_um),) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }
    positive_report = native_reports["positive"]
    hard_geometry_counts = {
        key: int(positive_report[key])
        for key in (
            "discontinuity_edge_count",
            "neighbor_wrap_risk_edge_count",
            "degenerate_quad_count",
            "folded_quad_count",
            "abrupt_normal_flip_edge_count",
            "distorted_quad_count",
        )
    }
    hard_geometry_pass = (
        all(value == 0 for value in hard_geometry_counts.values())
        and float(positive_report["in_bounds_vertex_fraction"]) == 1.0
        and float(positive_report["normal_valid_vertex_fraction"]) == 1.0
        and float(positive_report["offset_sample_in_bounds_fraction"]) == 1.0
    )

    sampler = PublicZarrChunkSampler(
        root_url=root,
        array_path=array_path,
        spec=spec,
        blosc_path=Path(args.blosc),
    )
    all_frames, all_sample_valid = sample_offsets(
        sampler,
        asset.surface,
        masks.normals_xyz,
        masks.sample_valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested_sample_mask = np.broadcast_to(
        masks.sample_valid, all_sample_valid.shape
    )
    sampling_complete = bool(np.all(all_sample_valid[requested_sample_mask]))
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset) for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support_frames = all_frames[support_indices]
    support_valid = all_sample_valid[support_indices]
    if not np.all(support_valid[np.broadcast_to(masks.sample_valid, support_valid.shape)]):
        sampling_complete = False
    support = support_summary(
        support_frames, DEFAULT_SUPPORT_OFFSETS, masks.sample_valid
    )
    transects = evaluate_valid_vertex_transects(
        all_frames,
        masks.sample_valid,
        seed=int(args.seed),
        sample_count=50,
    )
    exhaustive_transects = evaluate_all_vertex_transects(
        all_frames, masks.sample_valid
    )

    metadata_seed = asset.metadata.get("seed")
    seed_alignment: dict[str, Any] | None = None
    metadata_seed_m7: dict[str, Any] | None = None
    if isinstance(metadata_seed, (list, tuple)) and len(metadata_seed) == 3:
        seed_xyz = np.asarray(metadata_seed, dtype=np.float64)
        seed_alignment = seed_surface_alignment(
            asset.surface, seed_xyz, voxel_um=float(args.voxel_um)
        )
        seed_values, seed_valid = sampler.sample(
            seed_xyz[None, :], point_valid=np.ones(1, dtype=bool)
        )
        seed_quantized = np.rint(np.clip(seed_values, 0.0, 255.0)).astype(np.uint8)
        metadata_seed_m7 = {
            "xyz": seed_xyz.tolist(),
            "sample_valid": bool(seed_valid[0]),
            "trilinear_value": float(seed_values[0]),
            "quantized_value": int(seed_quantized[0]),
            "foreground": bool(seed_quantized[0] > 0),
        }

    generations_path = patch / "generations.tif"
    generations_report: dict[str, Any] | None = None
    generations_provenance: dict[str, Any] | None = None
    seed_support_fraction = support["support_fraction"]
    if generations_path.exists():
        generations = np.asarray(tifffile.imread(generations_path))
        generations_report = generation_support_summary(
            generations,
            masks.sample_valid,
            support_frames,
            DEFAULT_SUPPORT_OFFSETS,
            all_frames,
        )
        generations_provenance = {
            "path": str(generations_path.resolve()),
            "bytes": generations_path.stat().st_size,
            "sha256": sha256_file(generations_path),
            "array_content_sha256": _array_sha256(generations),
            "shape": list(generations.shape),
            "dtype": generations.dtype.str,
        }
        seed_support_fraction = float(
            generations_report["seed_core"]["support"]["support_fraction"]
        )

    go, gates = decide_intermediate_gate(
        hard_geometry_pass=hard_geometry_pass,
        sampling_complete=sampling_complete,
        support_fraction=float(support["support_fraction"]),
        minimum_support=float(args.minimum_support),
        seed_support_fraction=seed_support_fraction,
        minimum_seed_support=float(args.minimum_seed_support),
        transects_pass=bool(transects.get("pass", False)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "m7-quantized-offsets-minus15-plus15.npy"
    validity_path = output_dir / "m7-offset-sample-validity.npy"
    np.save(frames_path, all_frames, allow_pickle=False)
    np.save(validity_path, all_sample_valid, allow_pickle=False)
    result = {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": str(args.candidate_id),
        "verdict": "go_next_bounded_growth" if go else "no_go",
        "go": go,
        "verdict_scope": (
            "intermediate stored-grid geometry/m7 gate only; no ink inference; "
            "final full-resolution and self-intersection validation still required"
        ),
        "paid_compute_cost_usd": 0.0,
        "inputs": {
            "tifxyz": asset.manifest(),
            "generations": generations_provenance,
            "m7": {
                "root": root,
                "array_path": array_path,
                "group_metadata": group_provenance,
                "array_metadata": array_provenance,
                "root_attributes": attributes_provenance,
                "array_spec": asdict(spec),
                "codec": {
                    "path": str(Path(args.blosc).resolve()),
                    "sha256": sha256_file(Path(args.blosc)),
                },
            },
        },
        "algorithm_and_gates": {
            "coordinate_order": "TIFFXYZ xyz; m7 zyx",
            "surface_resolution": "stored 12x12 control grid",
            "normal_estimator": "native_surface_sampler mask-aware physical cross-gradient",
            "voxel_size_um": float(args.voxel_um),
            "quantization": "trilinear float -> rint clip[0,255] uint8 -> foreground >0",
            "support_offsets_voxels": list(DEFAULT_SUPPORT_OFFSETS),
            "transect_offsets_voxels": list(DEFAULT_TRANSECT_OFFSETS),
            "transect_rule": "exactly one foreground run containing offset 0",
            "transect_rng": "numpy.default_rng",
            "transect_rng_seed": int(args.seed),
            "minimum_support_fraction": float(args.minimum_support),
            "minimum_seed_core_support_fraction": float(args.minimum_seed_support),
            "gate_results": gates,
        },
        "geometry": {
            "hard_geometry_pass": hard_geometry_pass,
            "hard_geometry_counts": hard_geometry_counts,
            "localized_masks": masks.summary,
            "native_reports_by_orientation": native_reports,
            "metadata_seed_surface_alignment": seed_alignment,
            "self_intersection_status": "not evaluated; unknown, never assumed zero",
        },
        "m7": {
            "sampling_complete": sampling_complete,
            "metadata_seed_direct_sample": metadata_seed_m7,
            "support": support,
            "deterministic_50_transects": transects,
            "all_valid_vertex_transects_diagnostic": exhaustive_transects,
            "generation_support": generations_report,
            "quantized_frames": {
                "path": str(frames_path.resolve()),
                "sha256": sha256_file(frames_path),
                "array_content_sha256": _array_sha256(all_frames),
                "shape": list(all_frames.shape),
                "dtype": all_frames.dtype.str,
            },
            "sample_validity": {
                "path": str(validity_path.resolve()),
                "sha256": sha256_file(validity_path),
                "array_content_sha256": _array_sha256(all_sample_valid),
                "shape": list(all_sample_valid.shape),
                "dtype": all_sample_valid.dtype.str,
            },
            "public_zarr_sampling": sampler.manifest(),
        },
        "limitations": [
            "The patch is only 0.0147 cm2 and far below the final 0.5 cm2 minimum.",
            "Stored-grid checks do not localize or exclude self-intersection.",
            "A go verdict permits only another bounded geometry-growth step.",
            "No raw rendering, ink model, letter detector, or semantic inference was run.",
        ],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-id", default="priority1-seed-3525-4191-14071-growpatch-5gen"
    )
    parser.add_argument("--tifxyz", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="validation.json")
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--minimum-support", type=float, default=0.90)
    parser.add_argument("--minimum-seed-support", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1447)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output_dir) / str(args.output_name)
    try:
        if not 0 <= args.minimum_support <= 1:
            raise ValueError("--minimum-support must be in [0,1]")
        if not 0 <= args.minimum_seed_support <= 1:
            raise ValueError("--minimum-seed-support must be in [0,1]")
        result = run_validation(args)
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "error",
            "candidate_id": str(args.candidate_id),
            "verdict": "no_go",
            "go": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs": {
                "tifxyz": sanitize_source(args.tifxyz),
                "m7_root": sanitize_source(args.m7_root),
            },
            "paid_compute_cost_usd": 0.0,
        }
        _atomic_json(output_path, result)
        print(json.dumps(_json_safe(result), sort_keys=True), file=sys.stderr)
        return 1
    _atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "verdict": result["verdict"],
                "go": result["go"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if result["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
