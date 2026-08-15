#!/usr/bin/env python3
"""Plan and render the gated PHerc1203 surface from official public raw CT.

The exact full-resolution repaired surface is materialized and content-hashed
before any raw chunk access.  An immutable per-tile/per-chunk plan is then
written.  A sparse seven-offset preview is rendered first from a hash-sidecar
cache, followed by the 93 positive-normal layers (-46..+46).  Negative-normal
layers are an exact index reversal and are never duplicated.

No detector, OCR, semantic claim, submission, or paid compute is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from native_surface_sampler import (
    SurfaceGrid,
    estimate_surface_normals,
    validate_surface_geometry,
)
from PIL import Image
from scipy import ndimage

from render_pherc0800_rank2_raw_stack import (
    CachedPublicZarrArray,
    build_diagnostics,
    contact_sheet,
    model_window_mappings,
    save_png,
    scalable_topology_report,
    physical_area_cm2 as scalable_physical_area_cm2,
    window_u8,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import (
    ZarrV2ArraySpec,
    chunk_key,
    decode_chunk_payload,
    sha256_file,
)
from tifxyz_render_pipeline import (
    RenderOptions,
    _tile_roi_bounds_zyx,
    iter_surface_tiles,
    load_tifxyz_asset,
    render_surface_to_directory,
)
from validate_pherc0800_masked_candidate import quad_union_vertex_mask
from validate_pherc1203_auto_grown_candidate import (
    FULL_OFFSETS,
    PINNED_VOLUME_SHAPE_ZYX,
    VOXEL_UM,
    _safe_write_bytes,
    compact_full_geometry,
    derived_full_shape,
    full_geometry_pass,
    sha256_bytes,
    write_manifest,
)
from validate_public_m7_patch import _array_sha256, _atomic_json


CANDIDATE_ID = "auto_grown_20250925165041633"
DEFAULT_VALIDATION_ROOT = Path(
    "outputs/first-letters-geometry/pherc1203-auto-grown-candidate-validation"
) / CANDIDATE_ID
DEFAULT_SURFACE = DEFAULT_VALIDATION_ROOT / "materialized-topology-repaired-component"
DEFAULT_STORED_VALIDATION = DEFAULT_VALIDATION_ROOT / "stored-validation.json"
DEFAULT_FULL_VALIDATION = DEFAULT_VALIDATION_ROOT / "full-resolution-preflight.json"
DEFAULT_SELF_INTERSECTION_AUDIT = (
    DEFAULT_VALIDATION_ROOT / "self-intersection-audit.json"
)
PINNED_SELF_INTERSECTION_AUDIT_SHA256 = (
    "7c3b2b1899edf24b2f125cdc2ee070e6542a0011d32c872884f5e968c7d47a03"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/pherc1203-auto-grown-raw-stack-93"
)
DEFAULT_RAW_ROOT = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr"
)
DEFAULT_ARRAY_PATH = "0"
SPARSE_OFFSETS = (-46, -31, -15, 0, 15, 31, 46)
ALGORITHM_VERSION = "pherc1203-auto-grown-free-raw-render-v1"
RAW_USER_AGENT = "pherc1203-first-letters-free-raw-render/1"


def fetch_or_pin_json(
    url: str, path: Path, *, retries: int = 3
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use immutable local metadata after the first bounded-retry public GET."""

    source = "pinned_local_cache"
    if path.is_file():
        payload = path.read_bytes()
    else:
        source = "public_https_first_pin"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": RAW_USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                break
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                last_error = error
                if attempt + 1 == retries:
                    raise
                time.sleep(0.5 * (2**attempt))
        else:  # pragma: no cover
            raise RuntimeError("metadata retry loop exhausted") from last_error
        _safe_write_bytes(path, payload)
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("raw Zarr metadata must be a JSON object")
    return parsed, {
        "url": url,
        "path": str(path.resolve()),
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "source": source,
    }


def validate_raw_metadata(
    root: str, array_path: str, cache: Path
) -> tuple[ZarrV2ArraySpec, dict[str, Any]]:
    root = root.rstrip("/")
    group, group_record = fetch_or_pin_json(root + "/.zgroup", cache / "zgroup.json")
    attrs, attrs_record = fetch_or_pin_json(root + "/.zattrs", cache / "zattrs.json")
    array, array_record = fetch_or_pin_json(
        root + "/" + array_path.strip("/") + "/.zarray",
        cache / "array.zarray.json",
    )
    if group.get("zarr_format") != 2:
        raise ValueError("eligible raw root is not a Zarr-v2 group")
    axes = attrs.get("multiscales", [{}])[0].get("axes", [])
    names = [item.get("name") if isinstance(item, Mapping) else item for item in axes]
    if names != ["z", "y", "x"]:
        raise ValueError(f"eligible raw axes changed: {names!r}")
    spec = ZarrV2ArraySpec.from_metadata(array)
    if spec.shape_zyx != PINNED_VOLUME_SHAPE_ZYX:
        raise ValueError("eligible raw shape differs from the pinned 9.362um scan")
    if spec.dtype != np.dtype(np.uint8):
        raise ValueError(f"eligible raw dtype changed: {spec.dtype}")
    return spec, {
        "root": root,
        "array_path": array_path,
        "group": group_record,
        "root_attributes": attrs_record,
        "array_metadata": array_record,
        "axes": names,
        "shape_zyx": list(spec.shape_zyx),
        "chunks_zyx": list(spec.chunks_zyx),
        "dtype": spec.dtype.str,
        "order": spec.order,
        "dimension_separator": spec.dimension_separator,
        "compressor": spec.compressor,
        "filters": spec.filters,
    }


def full_surface_from_validations(
    surface_directory: Path,
    stored_validation_path: Path,
    full_validation_path: Path,
) -> tuple[Any, dict[str, Any]]:
    stored_payload = stored_validation_path.read_bytes()
    full_payload = full_validation_path.read_bytes()
    stored = json.loads(stored_payload)
    full = json.loads(full_payload)
    if stored.get("verdict") != "pass_to_full_resolution_preflight":
        raise ValueError("stored validation does not authorize raw planning")
    if full.get("verdict") != "pass_to_raw_ct_planning":
        raise ValueError("full validation does not authorize raw planning")
    if stored.get("candidate_id") != CANDIDATE_ID or full.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("validation candidate identity mismatch")
    asset = load_tifxyz_asset(
        surface_directory,
        resolution="full",
        interpolation="linear",
        maximum_surface_pixels=10_000_000,
    )
    if tuple(asset.output_shape) != derived_full_shape(asset.stored_shape, asset.scale_yx):
        raise ValueError("canonical full shape does not match metadata scale")
    initial_quads = (
        asset.surface.valid[:-1, :-1]
        & asset.surface.valid[:-1, 1:]
        & asset.surface.valid[1:, :-1]
        & asset.surface.valid[1:, 1:]
    )
    mask = quad_union_vertex_mask(initial_quads)
    cleaned = SurfaceGrid.from_tifxyz(
        asset.surface.x, asset.surface.y, asset.surface.z, mask=mask
    )
    asset = replace(
        asset,
        surface=cleaned,
        mask_source=asset.mask_source
        + "; strictly-subtractive-full-resolution-unused-vertex-cleanup",
    )
    expected_mask_record = full["explicit_full_resolution_valid_mask"]
    expected_mask_path = Path(str(expected_mask_record["path"]))
    if sha256_file(expected_mask_path) != expected_mask_record["sha256"]:
        raise ValueError("full validation mask file hash mismatch")
    expected_mask = tifffile.imread(expected_mask_path) != 0
    if not np.array_equal(expected_mask, asset.surface.valid):
        raise ValueError("reconstructed full surface mask differs from validated mask")
    return asset, {
        "stored_validation": {
            "path": str(stored_validation_path.resolve()),
            "sha256": sha256_bytes(stored_payload),
            "verdict": stored["verdict"],
        },
        "full_validation": {
            "path": str(full_validation_path.resolve()),
            "sha256": sha256_bytes(full_payload),
            "verdict": full["verdict"],
            "physical_area_cm2": full["physical_area_cm2"],
            "self_intersection_status": full["topology"][
                "self_intersection_status"
            ],
            "topology": full["topology"],
        },
        "validated_mask": expected_mask_record,
    }


def materialize_full_surface(
    asset: Any, destination: Path, lineage: Mapping[str, Any]
) -> tuple[dict[str, Any], Any]:
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {
        "x.tif": np.asarray(asset.surface.x, dtype=np.float32),
        "y.tif": np.asarray(asset.surface.y, dtype=np.float32),
        "z.tif": np.asarray(asset.surface.z, dtype=np.float32),
        "mask.tif": asset.surface.valid.astype(np.uint8) * 255,
    }
    records: list[dict[str, Any]] = []
    quantization: dict[str, Any] = {}
    for name, array in arrays.items():
        path = destination / name
        tifffile.imwrite(path, array, photometric="minisblack", compression="deflate")
        records.append(
            {
                "filename": name,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "array_content_sha256": _array_sha256(array),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            }
        )
        if name in {"x.tif", "y.tif", "z.tif"}:
            component = name[0]
            original = np.asarray(getattr(asset.surface, component), dtype=np.float64)
            delta = np.abs(original[asset.surface.valid] - array[asset.surface.valid])
            quantization[component] = {
                "maximum_absolute_delta_voxels": float(np.max(delta, initial=0.0)),
                "mean_absolute_delta_voxels": float(np.mean(delta)) if delta.size else 0.0,
                "serialization_dtype": "float32",
            }
    points = asset.surface.points_xyz[asset.surface.valid]
    meta = {
        "uuid": CANDIDATE_ID + "_canonical_full_resolution_repaired",
        "format": "tifxyz",
        "scale": [1.0, 1.0],
        "bbox": [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()],
        "source": "canonical_mask_aware_linear_resample_no_raw_ct_dependency",
        "lineage": dict(lineage),
    }
    meta_path = destination / "meta.json"
    _safe_write_bytes(
        meta_path,
        (json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    records.append(
        {
            "filename": "meta.json",
            "path": str(meta_path.resolve()),
            "sha256": sha256_file(meta_path),
            "bytes": meta_path.stat().st_size,
        }
    )
    reloaded = load_tifxyz_asset(destination, resolution="stored")
    for original, copied in zip(
        (asset.surface.x, asset.surface.y, asset.surface.z),
        (reloaded.surface.x, reloaded.surface.y, reloaded.surface.z),
        strict=True,
    ):
        if not np.array_equal(
            np.asarray(original, dtype=np.float32), copied, equal_nan=True
        ):
            raise RuntimeError("materialized full coordinate changed on reload")
    if not np.array_equal(reloaded.surface.valid, asset.surface.valid):
        raise RuntimeError("materialized full mask changed on reload")
    manifest = {
        "directory": str(destination.resolve()),
        "shape_yx": list(asset.surface.shape),
        "valid_vertex_count": int(np.count_nonzero(asset.surface.valid)),
        "coordinate_policy": "canonical linear resample; float32 materialization; no raw CT",
        "float64_to_float32_coordinate_quantization": quantization,
        "files": records,
        "reload_manifest": reloaded.manifest(),
    }
    return manifest, reloaded


def validate_reloaded_float32_surface(
    asset: Any, validation: Mapping[str, Any]
) -> dict[str, Any]:
    """Repeat the final topology/area/native gate on serialized coordinates."""

    topology = scalable_topology_report(asset.surface.valid)
    reference_topology = validation["full_validation"]["topology"]
    topology_identical = topology == reference_topology
    area = scalable_physical_area_cm2(asset.surface, voxel_um=VOXEL_UM)
    reference_area = float(validation["full_validation"]["physical_area_cm2"])
    area_delta = area - reference_area
    # A 0.01% ceiling is declared before inspection and is much smaller than
    # the 0.5 cm2 prize threshold margin.  The exact observed delta is retained.
    maximum_area_delta = max(1e-8, abs(reference_area) * 1e-4)
    offsets_um = tuple(float(value) * VOXEL_UM for value in FULL_OFFSETS)
    reports: dict[str, Any] = {}
    for name, sign in (("positive", 1), ("negative", -1)):
        raw = validate_surface_geometry(
            asset.surface,
            volume_shape_zyx=PINNED_VOLUME_SHAPE_ZYX,
            voxel_size_zyx_um=(VOXEL_UM,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        reports[name] = {**compact_full_geometry(raw), "pass": full_geometry_pass(raw)}
    gates = [
        {
            "name": "reloaded_mask_topology_bit_identical_to_pre_serialization_gate",
            "pass": topology_identical,
        },
        {
            "name": "reloaded_float32_area_delta_within_predeclared_ceiling",
            "value_cm2": area_delta,
            "absolute_value_cm2": abs(area_delta),
            "maximum_absolute_delta_cm2": maximum_area_delta,
            "pass": abs(area_delta) <= maximum_area_delta,
        },
        {
            "name": "reloaded_float32_area_still_above_prize_surface_minimum",
            "value_cm2": area,
            "threshold_cm2": 0.5,
            "pass": area >= 0.5,
        },
        {
            "name": "reloaded_float32_positive_native_qc_minus46_plus46",
            "pass": reports["positive"]["pass"],
        },
        {
            "name": "reloaded_float32_negative_native_qc_minus46_plus46",
            "pass": reports["negative"]["pass"],
        },
    ]
    return {
        "status": "complete_before_raw_metadata_or_chunk_access",
        "coordinate_source": "reloaded full-resolution x/y/z/mask TIFFXYZ float32",
        "topology": topology,
        "topology_identical_to_pre_serialization_gate": topology_identical,
        "reference_area_cm2": reference_area,
        "reloaded_float32_area_cm2": area,
        "area_delta_cm2": area_delta,
        "maximum_allowed_absolute_area_delta_cm2": maximum_area_delta,
        "normal_offset_voxels": list(FULL_OFFSETS),
        "native_geometry_by_sign": reports,
        "gate_results": gates,
        "pass": all(bool(gate["pass"]) for gate in gates),
        "self_intersection_status": "unknown_not_computed",
    }


def bind_self_intersection_audit(
    source: Path, destination: Path, validation: Mapping[str, Any]
) -> dict[str, Any]:
    payload = source.read_bytes()
    digest = sha256_bytes(payload)
    if digest != PINNED_SELF_INTERSECTION_AUDIT_SHA256:
        raise ValueError("pinned self-intersection audit hash changed")
    report = json.loads(payload)
    if report.get("status") != "complete" or report.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("self-intersection audit identity/status mismatch")
    if (
        report["inputs"]["stored_validation"]["sha256"]
        != validation["stored_validation"]["sha256"]
        or report["inputs"]["full_resolution_preflight"]["sha256"]
        != validation["full_validation"]["sha256"]
    ):
        raise ValueError("self-intersection audit was run on different validations")
    if int(
        report["stored_exact_audit"]["narrow_phase"][
            "total_non_adjacent_intersection_count"
        ]
    ) != 0:
        raise ValueError("self-intersection audit did not certify zero stored crossings")
    clearance = float(
        report["canonical_linear_full_raster"][
            "certified_non_adjacent_full_raster_clearance_lower_bound_voxels"
        ]
    )
    if clearance <= 0.0:
        raise ValueError("self-intersection audit has no positive full-raster bound")
    local = destination / "self-intersection-audit.json"
    _safe_write_bytes(local, payload)
    if sha256_file(local) != digest:
        raise RuntimeError("bound self-intersection evidence copy hash mismatch")
    if report["campaign_gate"]["raw_ct_diagnostic"] != "allow":
        raise ValueError("self-intersection campaign gate does not allow raw diagnostic")
    return {
        "source_path": str(source.resolve()),
        "bound_copy_path": str(local.resolve()),
        "sha256": digest,
        "scope": report["intersection_semantics"]["pair_scope"],
        "stored_total_non_adjacent_intersection_count": 0,
        "stored_minimum_nonadjacent_clearance_voxels": report[
            "near_touch_diagnostic"
        ]["minimum_distance_voxels"],
        "stored_minimum_nonadjacent_clearance_micrometers": report[
            "near_touch_diagnostic"
        ]["minimum_distance_micrometers"],
        "certified_nonadjacent_full_raster_clearance_lower_bound_voxels": clearance,
        "certified_nonadjacent_full_raster_clearance_lower_bound_micrometers": report[
            "canonical_linear_full_raster"
        ]["certified_non_adjacent_full_raster_clearance_lower_bound_micrometers"],
        "bilinear_corner_jacobian_check": report["canonical_linear_full_raster"][
            "local_patch_fold_check"
        ],
        "precise_status": (
            "stored canonical triangle pairs sharing no stored-grid vertex: exact zero "
            "intersections; canonical full-raster coarse-topology-nonadjacent patch "
            "crossings excluded by positive clearance bound; shared-simplex pairs were "
            "excluded and same/immediately topology-adjacent coarse patch neighborhoods "
            "remain not exhaustively exact-tested"
        ),
        "raw_ct_diagnostic": "allow",
        "prize_submission": report["campaign_gate"]["prize_submission"],
    }


def chunk_indices_for_roi(
    spec: ZarrV2ArraySpec, roi: tuple[slice, slice, slice]
) -> list[tuple[int, int, int]]:
    ranges = []
    for selection, chunk in zip(roi, spec.chunks_zyx, strict=True):
        first = int(selection.start) // int(chunk)
        last = (int(selection.stop) - 1) // int(chunk)
        ranges.append(range(first, last + 1))
    return [
        (z, y, x) for z in ranges[0] for y in ranges[1] for x in ranges[2]
    ]


def chunk_shape(spec: ZarrV2ArraySpec, index: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(
        min(chunk, size - coordinate * chunk)
        for size, chunk, coordinate in zip(
            spec.shape_zyx, spec.chunks_zyx, index, strict=True
        )
    )  # type: ignore[return-value]


def build_chunk_plan(
    asset: Any,
    spec: ZarrV2ArraySpec,
    root: str,
    array_path: str,
    *,
    tile_size: int,
) -> dict[str, Any]:
    normals = estimate_surface_normals(
        asset.surface, voxel_size_zyx_um=(VOXEL_UM,) * 3
    )
    offsets_um = tuple(float(value) * VOXEL_UM for value in FULL_OFFSETS)
    tiles: list[dict[str, Any]] = []
    all_chunks: set[tuple[int, int, int]] = set()
    for tile_index, (rows, columns) in enumerate(
        iter_surface_tiles(asset.surface.shape, (tile_size, tile_size))
    ):
        key = (rows, columns)
        base = np.stack(
            (
                asset.surface.x[key],
                asset.surface.y[key],
                asset.surface.z[key],
            ),
            axis=-1,
        )
        valid = asset.surface.valid[key] & normals.valid[key]
        roi = _tile_roi_bounds_zyx(
            base,
            normals.positive_xyz[key],
            valid,
            volume_shape_zyx=spec.shape_zyx,
            voxel_size_xyz_um=np.asarray((VOXEL_UM,) * 3)[::-1],
            offsets_um=offsets_um,
            signs=("positive",),
        )
        if roi is None:
            chunks: list[tuple[int, int, int]] = []
            roi_record = None
        else:
            chunks = chunk_indices_for_roi(spec, roi)
            roi_record = [
                [int(selection.start), int(selection.stop)] for selection in roi
            ]
            all_chunks.update(chunks)
        tiles.append(
            {
                "tile_index": tile_index,
                "rows_half_open": [int(rows.start), int(rows.stop)],
                "columns_half_open": [int(columns.start), int(columns.stop)],
                "roi_zyx_half_open": roi_record,
                "chunk_indices_zyx": [list(item) for item in chunks],
            }
        )
    del normals
    chunks = sorted(all_chunks)
    decoded_upper = sum(
        math.prod(chunk_shape(spec, item)) * spec.dtype.itemsize for item in chunks
    )
    root = root.rstrip("/")
    return {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete_before_raw_chunk_access",
        "candidate_id": CANDIDATE_ID,
        "surface_shape_yx": list(asset.surface.shape),
        "surface_mask_content_sha256": _array_sha256(asset.surface.valid),
        "tile_shape_yx": [tile_size, tile_size],
        "offsets_voxels": list(FULL_OFFSETS),
        "normal_signs_materialized": ["positive"],
        "negative_equivalence": "positive layer index reversal; not duplicated",
        "raw_root": root,
        "array_path": array_path,
        "raw_shape_zyx": list(spec.shape_zyx),
        "raw_chunks_zyx": list(spec.chunks_zyx),
        "tile_count": len(tiles),
        "unique_chunk_count": len(chunks),
        "decoded_chunk_bytes_upper_bound": decoded_upper,
        "unique_chunks": [
            {
                "chunk_index_zyx": list(item),
                "chunk_key": chunk_key(spec, item),
                "expected_decoded_shape_zyx": list(chunk_shape(spec, item)),
                "url": root
                + "/"
                + array_path.strip("/")
                + "/"
                + chunk_key(spec, item),
            }
            for item in chunks
        ],
        "tiles": tiles,
    }


class PlannedHashedPublicZarrArray(CachedPublicZarrArray):
    def __init__(self, *, allowed_chunks: set[tuple[int, int, int]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.allowed_chunks = allowed_chunks

    def _fetch_public(self, url: str) -> bytes:
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": RAW_USER_AGENT})
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read()
            except (OSError, urllib.error.URLError, urllib.error.HTTPError):
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2**attempt))
        raise AssertionError("unreachable")

    def _load_chunk(self, index, expected_shape):
        if index not in self.allowed_chunks:
            raise RuntimeError(f"renderer requested unplanned raw chunk: {index}")
        payload_path, missing_path = self._paths(index)
        sidecar = payload_path.with_suffix(payload_path.suffix + ".sha256")
        if payload_path.exists() and sidecar.exists():
            expected = sidecar.read_text(encoding="ascii").strip()
            if sha256_file(payload_path) != expected:
                raise RuntimeError(f"raw chunk cache hash mismatch: {index}")
        if missing_path.exists() and missing_path.read_bytes() != b"missing public Zarr chunk\n":
            raise RuntimeError(f"raw missing marker changed: {index}")
        array, provenance = super()._load_chunk(index, expected_shape)
        if payload_path.exists():
            digest = sha256_file(payload_path)
            record = self.records[index]
            if record.get("payload_sha256") != digest:
                raise RuntimeError(f"decoded raw payload hash mismatch: {index}")
            if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != digest:
                raise RuntimeError(f"raw hash sidecar mismatch after read: {index}")
            _safe_write_bytes(sidecar, (digest + "\n").encode("ascii"))
        return array, provenance


def verify_cache_scope(cache: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        tuple(int(value) for value in item["chunk_index_zyx"])
        for item in plan["unique_chunks"]
    }


def prefetch_planned_chunks(
    plan: Mapping[str, Any],
    spec: ZarrV2ArraySpec,
    cache: Path,
    blosc: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Concurrently fill only the immutable plan, validating before commit."""

    items = list(plan["unique_chunks"])

    def fetch_one(item: Mapping[str, Any]) -> dict[str, Any]:
        index = tuple(int(value) for value in item["chunk_index_zyx"])
        expected_shape = tuple(
            int(value) for value in item["expected_decoded_shape_zyx"]
        )
        payload_path = cache / str(index[0]) / str(index[1]) / f"{index[2]}.bin"
        missing_path = cache / str(index[0]) / str(index[1]) / f"{index[2]}.missing"
        sidecar = payload_path.with_suffix(payload_path.suffix + ".sha256")
        network_fetch = False
        if missing_path.exists():
            if missing_path.read_bytes() != b"missing public Zarr chunk\n":
                raise RuntimeError(f"raw missing marker changed: {index}")
            return {
                "chunk_index_zyx": list(index),
                "status": "verified_cached_missing_fill_value",
                "url": item["url"],
                "network_fetch": False,
            }
        if payload_path.exists():
            payload = payload_path.read_bytes()
            digest = sha256_bytes(payload)
            if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != digest:
                raise RuntimeError(f"raw payload sidecar mismatch: {index}")
            status = "verified_payload_cache_hit"
        else:
            payload = b""
            for attempt in range(3):
                try:
                    request = urllib.request.Request(
                        str(item["url"]), headers={"User-Agent": RAW_USER_AGENT}
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        payload = response.read()
                    break
                except urllib.error.HTTPError as error:
                    if error.code == 404:
                        _safe_write_bytes(
                            missing_path, b"missing public Zarr chunk\n"
                        )
                        return {
                            "chunk_index_zyx": list(index),
                            "status": "network_404_fill_value",
                            "url": item["url"],
                            "network_fetch": True,
                        }
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (2**attempt))
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (2**attempt))
            digest = sha256_bytes(payload)
            status = "network_downloaded_verified"
            network_fetch = True
        _array, decoder, decoded = decode_chunk_payload(
            payload,
            expected_shape,
            spec,
            blosc_path=blosc,
        )
        if not payload_path.exists():
            _safe_write_bytes(payload_path, payload)
        _safe_write_bytes(sidecar, (digest + "\n").encode("ascii"))
        if sha256_file(payload_path) != digest:
            raise RuntimeError(f"raw payload changed after cache commit: {index}")
        return {
            "chunk_index_zyx": list(index),
            "status": status,
            "url": item["url"],
            "payload_path": str(payload_path.resolve()),
            "payload_bytes": len(payload),
            "payload_sha256": digest,
            "decoded_bytes": len(decoded),
            "decoded_sha256": sha256_bytes(decoded),
            "decoder": decoder,
            "network_fetch": network_fetch,
        }

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, item): item for item in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed == 1 or completed % 25 == 0 or completed == len(items):
                print(
                    json.dumps(
                        {
                            "stage": "raw_chunk_prefetch",
                            "complete": completed,
                            "total": len(items),
                        }
                    ),
                    flush=True,
                )
    records.sort(key=lambda value: value["chunk_index_zyx"])
    return {
        "schema_version": 1,
        "status": "complete",
        "plan_sha256": str(plan.get("plan_sha256", "recorded_separately")),
        "worker_count": workers,
        "elapsed_seconds": time.monotonic() - started,
        "record_count": len(records),
        "network_fetch_count": sum(bool(item["network_fetch"]) for item in records),
        "network_payload_bytes": sum(
            int(item.get("payload_bytes", 0))
            for item in records
            if item["network_fetch"]
        ),
        "verified_payload_count": sum("payload_sha256" in item for item in records),
        "missing_fill_value_count": sum(
            "missing" in str(item["status"]) for item in records
        ),
        "records": records,
    }
    unexpected: list[str] = []
    verified = 0
    verified_bytes = 0
    for path in cache.rglob("*.bin") if cache.exists() else ():
        relative = path.relative_to(cache).parts
        if len(relative) != 3:
            unexpected.append(str(path))
            continue
        index = (int(relative[0]), int(relative[1]), int(Path(relative[2]).stem))
        if index not in allowed:
            unexpected.append(str(path))
            continue
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.is_file() and sidecar.read_text(encoding="ascii").strip() != sha256_file(path):
            raise RuntimeError(f"preexisting raw cache hash mismatch: {index}")
        verified += 1
        verified_bytes += path.stat().st_size
    if unexpected:
        raise RuntimeError("dedicated raw cache contains chunks outside immutable plan")
    return {
        "allowed_chunk_count": len(allowed),
        "preexisting_verified_payload_count": verified,
        "preexisting_verified_payload_bytes": verified_bytes,
        "unexpected_payload_count": 0,
    }


def sparse_diagnostics(stack: Path, output: Path, explicit_mask: np.ndarray) -> dict[str, Any]:
    directory = stack / "positive"
    paths = [directory / f"{index:02d}.tif" for index in range(len(SPARSE_OFFSETS))]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("sparse raw renderer did not produce all seven layers")
    valid = tifffile.imread(directory / "valid-all.tif") != 0
    if not np.array_equal(valid, explicit_mask):
        raise RuntimeError("sparse renderer valid-all mask differs from explicit mask")
    frames = [np.asarray(tifffile.memmap(path), dtype=np.uint8) for path in paths]
    numeric = np.stack(frames, axis=0).astype(np.float32)
    center = numeric[3]
    mean = np.mean(numeric, axis=0)
    depth_range = np.max(numeric, axis=0) - np.min(numeric, axis=0)
    residual = center - ndimage.gaussian_filter(center, sigma=8.0, mode="nearest")
    diagnostic_dir = output / "early-diagnostics"
    records: dict[str, Any] = {}
    for name, array, symmetric in (
        ("center_offset0", center, False),
        ("sparse_depth_mean", mean, False),
        ("sparse_depth_range", depth_range, False),
        ("center_local_residual", residual, True),
    ):
        image, window = window_u8(array, valid, symmetric=symmetric)
        records[name] = {
            **save_png(diagnostic_dir / f"{name}.png", image),
            "display_window": window,
        }
    contact = []
    for offset, frame in zip(SPARSE_OFFSETS, frames, strict=True):
        image, _window = window_u8(frame, valid)
        contact.append((f"offset {offset:+d} vox", image))
    sheet = contact_sheet(contact)
    contact_path = diagnostic_dir / "sparse-seven-offset-contact-sheet.png"
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_path, optimize=True)
    records["contact_sheet"] = {
        "path": str(contact_path.resolve()),
        "sha256": sha256_file(contact_path),
        "offsets_voxels": list(SPARSE_OFFSETS),
    }
    return {
        "status": "complete_before_full_93_layer_render",
        "role": "visibility aids only; no letter or ink claim",
        "valid_mask_identity": True,
        "records": records,
    }


def verify_full_stack(stack: Path, explicit_mask: np.ndarray) -> dict[str, Any]:
    directory = stack / "positive"
    layer_paths = [directory / f"{index:02d}.tif" for index in range(93)]
    if not all(path.is_file() for path in layer_paths):
        raise FileNotFoundError("full raw renderer did not produce all 93 layers")
    valid_all = tifffile.imread(directory / "valid-all.tif") != 0
    if not np.array_equal(valid_all, explicit_mask):
        raise RuntimeError("full renderer valid-all mask differs from explicit mask")
    outside_nonzero: list[int] = []
    records = []
    for index, path in enumerate(layer_paths):
        layer = tifffile.memmap(path)
        if np.any(layer[~explicit_mask] != 0):
            outside_nonzero.append(index)
        records.append(
            {
                "frame": index,
                "offset_voxels": int(FULL_OFFSETS[index]),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    if outside_nonzero:
        raise RuntimeError(f"raw layers are nonzero outside explicit mask: {outside_nonzero}")
    return {
        "layer_count": 93,
        "shape_frame_row_column": [93, *explicit_mask.shape],
        "dtype": "uint8",
        "valid_all_mask_identity": True,
        "zero_outside_explicit_mask_all_layers": True,
        "negative_stack_policy": "not stored; exact layer reversal 92..0",
        "layers": records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    asset, validation = full_surface_from_validations(
        Path(args.surface),
        Path(args.stored_validation),
        Path(args.full_validation),
    )
    materialized, reloaded_asset = materialize_full_surface(
        asset,
        output / "full-resolution-surface",
        lineage=validation,
    )
    float32_gate = validate_reloaded_float32_surface(reloaded_asset, validation)
    float32_gate_path = output / "full-resolution-float32-validation.json"
    _atomic_json(float32_gate_path, float32_gate)
    if not float32_gate["pass"]:
        raise RuntimeError("reloaded float32 full-resolution surface gate failed")
    # From this point onward, planning and rendering use only the exact reloaded
    # float32 coordinates and explicit mask that passed the superseding gate.
    asset = reloaded_asset
    materialized["reloaded_float32_validation"] = {
        "path": str(float32_gate_path.resolve()),
        "sha256": sha256_file(float32_gate_path),
        "pass": True,
        "area_delta_cm2": float32_gate["area_delta_cm2"],
    }
    materialized_manifest_path = output / "full-resolution-surface-manifest.json"
    _atomic_json(materialized_manifest_path, materialized)

    self_intersection = bind_self_intersection_audit(
        Path(args.self_intersection_audit),
        output / "bound-geometry-evidence",
        validation,
    )

    spec, metadata = validate_raw_metadata(
        str(args.raw_root), str(args.array_path), output / "raw-metadata-cache"
    )
    plan = build_chunk_plan(
        asset,
        spec,
        str(args.raw_root),
        str(args.array_path),
        tile_size=int(args.tile_size),
    )
    plan_path = output / "raw-chunk-plan.json"
    _atomic_json(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    raw_cache = Path(args.raw_cache)
    cache_preflight = verify_cache_scope(raw_cache, plan)
    free_bytes = shutil.disk_usage(output).free
    plan_record = {
        "path": str(plan_path.resolve()),
        "sha256": plan_sha,
        "written_before_raw_chunk_access": True,
        "free_disk_bytes_at_plan_commit": free_bytes,
        "cache_preflight": cache_preflight,
    }
    _atomic_json(output / "raw-access-precommit.json", plan_record)
    if args.plan_only:
        return {
            "schema_version": 1,
            "status": "plan_complete",
            "verdict": "ready_for_free_raw_render",
            "paid_compute_cost_usd": 0.0,
            "chunk_plan": plan_record,
            "unique_chunk_count": plan["unique_chunk_count"],
            "decoded_chunk_bytes_upper_bound": plan[
                "decoded_chunk_bytes_upper_bound"
            ],
            "self_intersection_evidence": self_intersection,
        }

    blosc = Path(args.blosc)
    if not blosc.is_file():
        raise FileNotFoundError("pinned Blosc codec is missing")
    prefetch = prefetch_planned_chunks(
        {**plan, "plan_sha256": plan_sha},
        spec,
        raw_cache,
        blosc,
        workers=int(args.prefetch_workers),
    )
    prefetch_path = output / "raw-prefetch-manifest.json"
    _atomic_json(prefetch_path, prefetch)
    allowed = {
        tuple(int(value) for value in item["chunk_index_zyx"])
        for item in plan["unique_chunks"]
    }
    volume = PlannedHashedPublicZarrArray(
        root_url=str(args.raw_root),
        array_path=str(args.array_path),
        spec=spec,
        cache_directory=raw_cache,
        blosc_path=blosc,
        decoded_lru_chunks=int(args.decoded_lru_chunks),
        allowed_chunks=allowed,
    )

    sparse_stack = output / "raw-sparse-seven"
    sparse_options = RenderOptions(
        voxel_size_zyx_um=(VOXEL_UM,) * 3,
        offsets_um=tuple(float(value) * VOXEL_UM for value in SPARSE_OFFSETS),
        signs=("positive",),
        tile_shape=(int(args.tile_size), int(args.tile_size)),
        output_dtype="uint8",
        fill_value=0.0,
        png_mode="preview",
        overwrite=bool(args.overwrite),
        hash_outputs=True,
    )
    sparse_manifest = render_surface_to_directory(
        volume,
        asset.surface,
        sparse_stack,
        options=sparse_options,
        tifxyz_manifest=materialized,
        volume_manifest=metadata,
        progress=lambda index, total, detail: print(
            json.dumps({"stage": "sparse", "tile": index, "tiles": total, **detail}),
            flush=True,
        ),
    )
    early = sparse_diagnostics(
        sparse_stack, output, asset.surface.valid
    )
    _atomic_json(output / "early-diagnostics.json", early)

    full_stack = output / "raw-surface-stack"
    full_options = RenderOptions(
        voxel_size_zyx_um=(VOXEL_UM,) * 3,
        offsets_um=tuple(float(value) * VOXEL_UM for value in FULL_OFFSETS),
        signs=("positive",),
        tile_shape=(int(args.tile_size), int(args.tile_size)),
        output_dtype="uint8",
        fill_value=0.0,
        png_mode="preview",
        overwrite=bool(args.overwrite),
        hash_outputs=True,
    )
    full_manifest = render_surface_to_directory(
        volume,
        asset.surface,
        full_stack,
        options=full_options,
        tifxyz_manifest=materialized,
        volume_manifest=metadata,
        progress=lambda index, total, detail: print(
            json.dumps({"stage": "full", "tile": index, "tiles": total, **detail}),
            flush=True,
        ),
    )
    stack_verification = verify_full_stack(full_stack, asset.surface.valid)
    diagnostics = build_diagnostics(full_stack, output)
    cache_manifest = volume.manifest()
    if set(volume.records) - allowed:
        raise RuntimeError("raw reader accessed a chunk outside the immutable plan")
    mappings = model_window_mappings()
    mappings["stored_frame_to_offset_micrometers"] = {
        str(index): float(value) * VOXEL_UM
        for index, value in enumerate(FULL_OFFSETS)
    }
    final = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete",
        "verdict": "model_ready_free_raw_stack",
        "candidate_id": CANDIDATE_ID,
        "paid_compute_cost_usd": 0.0,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "validation": validation,
        "full_resolution_surface": materialized,
        "reloaded_float32_surface_validation": {
            "path": str(float32_gate_path.resolve()),
            "sha256": sha256_file(float32_gate_path),
            "report": float32_gate,
        },
        "self_intersection_evidence": self_intersection,
        "raw_public_zarr": {
            "metadata": metadata,
            "chunk_plan": plan_record,
            "prefetch": {
                "path": str(prefetch_path.resolve()),
                "sha256": sha256_file(prefetch_path),
                "record_count": prefetch["record_count"],
                "network_fetch_count": prefetch["network_fetch_count"],
                "network_payload_bytes": prefetch["network_payload_bytes"],
            },
            "chunk_reader": cache_manifest,
            "codec_path": str(blosc.resolve()),
            "codec_sha256": sha256_file(blosc),
        },
        "sparse_render": {
            "offsets_voxels": list(SPARSE_OFFSETS),
            "renderer": sparse_manifest,
            "early_diagnostics": early,
        },
        "raw_stack": {
            **stack_verification,
            "directory": str((full_stack / "positive").resolve()),
            "offsets_voxels": list(FULL_OFFSETS),
            "offsets_micrometers": [float(value) * VOXEL_UM for value in FULL_OFFSETS],
            "interpolation": "trilinear official raw volume sampling",
            "surface_resolution": "canonical full linear TIFFXYZ raster",
            "renderer": full_manifest,
        },
        "sign_order_and_model_window_mappings": mappings,
        "diagnostics": diagnostics,
        "self_intersection_status": self_intersection["precise_status"],
        "limitations": [
            "Shared-simplex pairs and same/immediately topology-adjacent coarse patch neighborhoods remain outside the exact nonadjacent-pair certificate.",
            "The bound audit allows raw diagnostics but still holds prize submission pending local-adjacency closure and candidate-specific text evidence.",
            "No detector, ink model, OCR, or semantic text inference was run.",
            "Direct-ink diagnostics are visibility aids, not evidence of letters.",
            "Only the positive stack is stored; the negative stack is exact layer reversal.",
        ],
    }
    final_path = output / "raw-stack-manifest.json"
    _atomic_json(final_path, final)
    manifest = write_manifest(output)
    return {
        **final,
        "artifact_manifest": manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    parser.add_argument("--stored-validation", type=Path, default=DEFAULT_STORED_VALIDATION)
    parser.add_argument("--full-validation", type=Path, default=DEFAULT_FULL_VALIDATION)
    parser.add_argument(
        "--self-intersection-audit",
        type=Path,
        default=DEFAULT_SELF_INTERSECTION_AUDIT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--array-path", default=DEFAULT_ARRAY_PATH)
    parser.add_argument("--raw-cache", type=Path, default=DEFAULT_OUTPUT / "raw-chunk-cache")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--decoded-lru-chunks", type=int, default=32)
    parser.add_argument("--prefetch-workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "error",
            "candidate_id": CANDIDATE_ID,
            "verdict": "render_error",
            "paid_compute_cost_usd": 0.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        evidence = Path(args.output) / f"raw-render-attempt-error-{int(time.time())}.json"
        _atomic_json(evidence, failure)
        print(json.dumps({**failure, "attempt_evidence": str(evidence)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["verdict"],
                "output": str(Path(args.output).resolve()),
                "unique_chunk_count": result.get("unique_chunk_count"),
                "artifact_manifest": result.get("artifact_manifest"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
