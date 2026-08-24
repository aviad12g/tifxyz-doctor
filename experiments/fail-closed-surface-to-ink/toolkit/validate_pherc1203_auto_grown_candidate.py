#!/usr/bin/env python3
"""Fail-closed PHerc1203 stored-grid m7 and full-raster geometry validation.

The candidate component is fixed by the completed 22-surface preflight.  This
program copies source coordinates byte-for-byte, recomputes boundary normals,
selects the fresh centered component, applies the existing strictly-subtractive
topology repair, and repeats the public-m7 validation after repair.  Only a
passing stored-grid result is canonically resampled to the metadata-derived
full raster for both-sign native geometry validation at offsets -46..+46.

No raw CT, renderer, ink model, detector, OCR, or paid compute is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from native_surface_sampler import SurfaceGrid, validate_surface_geometry
from numpy.typing import NDArray

from audit_pherc0800_published_segments import (
    DiskPayloadFetcher,
    connected_quad_components_summary,
    quad_component_vertex_mask,
    run_metrics_summary,
    selected_region_diagnostics,
    selected_run_metrics,
)
from preflight_debug_m7 import derive_geometry_masks, sample_offsets
from render_pherc0800_rank2_raw_stack import (
    active_quads as scalable_active_quads,
    physical_area_cm2 as scalable_physical_area_cm2,
    quad_union_vertex_mask as scalable_quad_union_vertex_mask,
    scalable_topology_report,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import ZarrV2ArraySpec, sha256_file
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_pherc0800_masked_candidate import (
    active_quads,
    conservative_manifold_subset,
    materialize_masked_tifxyz,
    quad_union_vertex_mask,
    save_array,
    topology_report,
)
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    _array_sha256,
    _atomic_json,
    support_summary,
    validate_metadata_axes,
)


CANDIDATE_ID = "auto_grown_20250925165041633"
FALLBACK_ID = "auto_grown_20250923163331524"
DEFAULT_PREFLIGHT = Path(
    "outputs/first-letters-geometry/"
    "pherc1203-auto-grown-stored-grid-m7-preflight/"
    "pherc1203-auto-grown-stored-grid-m7-preflight.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/first-letters-geometry/pherc1203-auto-grown-candidate-validation"
)
DEFAULT_M7_CACHE = DEFAULT_PREFLIGHT.parent / "m7-chunk-cache"
VOXEL_UM = 9.362
PINNED_VOLUME_SHAPE_ZYX = (18977, 6844, 6844)
STORED_OFFSETS = tuple(range(-15, 16))
SUPPORT_OFFSETS = (-2, -1, 0, 1, 2)
FULL_OFFSETS = tuple(range(-46, 47))
MINIMUM_AREA_CM2 = 0.5
ALGORITHM_VERSION = "pherc1203-auto-grown-masked-full-gate-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(root: Path) -> dict[str, Any]:
    path = root / "MANIFEST.sha256"
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    )
    rows = [f"{sha256_file(candidate)}  {candidate.relative_to(root)}" for candidate in files]
    _safe_write_bytes(path, ("\n".join(rows) + "\n").encode("utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_count": len(files),
        "total_bytes_excluding_manifest": sum(item.stat().st_size for item in files),
    }


def fetch_cached_json(
    url: str,
    cache_path: Path,
    *,
    expected_sha256: str,
    retries: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load pinned JSON from a verified cache or bounded-retry public GET."""

    source = "verified_local_cache"
    if cache_path.is_file() and sha256_file(cache_path) == expected_sha256:
        payload = cache_path.read_bytes()
    else:
        source = "public_https"
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "pherc1203-first-letters-gate/1"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                break
            except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
                last_error = error
                if attempt + 1 == retries:
                    raise
                time.sleep(0.5 * (2**attempt))
        else:  # pragma: no cover - loop either breaks or raises
            raise RuntimeError("metadata retry loop exhausted") from last_error
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError(f"pinned metadata hash mismatch: {url}")
        _safe_write_bytes(cache_path, payload)
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError(f"cached metadata hash mismatch: {cache_path}")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("public metadata is not a JSON object")
    return parsed, {
        "url": url,
        "cache_path": str(cache_path.resolve()),
        "sha256": expected_sha256,
        "bytes": len(payload),
        "source": source,
    }


def one_segment(report: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    matches = [
        item for item in report["audited_segments"] if item["long_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("preflight must contain exactly one requested candidate")
    return matches[0]


def load_verified_array(record: Mapping[str, Any]) -> NDArray[Any]:
    path = Path(str(record["path"]))
    if not path.is_file() or sha256_file(path) != record["file_sha256"]:
        raise ValueError(f"preflight array hash mismatch: {path}")
    array = np.load(path, allow_pickle=False)
    if _array_sha256(array) != record["array_content_sha256"]:
        raise ValueError(f"preflight array content hash mismatch: {path}")
    return array


def verify_source(segment: Mapping[str, Any]) -> Path:
    source = Path(str(segment["tifxyz"]["source"]))
    for filename, expected in segment["tifxyz"]["input_sha256"].items():
        path = source / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source TIFFXYZ hash mismatch: {path}")
    return source


def compact_stored_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    positive = geometry["native_reports_by_orientation"]["positive"]
    return {
        "pass": bool(geometry["pass"]),
        "hard_defect_counts": geometry["hard_defect_counts"],
        "valid_vertex_count": int(positive["valid_vertex_count"]),
        "valid_quad_count": int(positive["valid_quad_count"]),
        "normal_valid_vertex_fraction": float(positive["normal_valid_vertex_fraction"]),
        "in_bounds_vertex_fraction": float(positive["in_bounds_vertex_fraction"]),
        "offset_sample_in_bounds_fraction": float(
            positive["offset_sample_in_bounds_fraction"]
        ),
        "self_intersection_status": geometry["self_intersection_status"],
        "native_reports_by_orientation": geometry["native_reports_by_orientation"],
    }


def evaluate_stored_surface(
    surface: SurfaceGrid,
    sampler: PublicZarrChunkSampler,
    spec: ZarrV2ArraySpec,
) -> dict[str, Any]:
    masks = derive_geometry_masks(
        surface,
        volume_shape_zyx=spec.shape_zyx,
        voxel_um=VOXEL_UM,
        maximum_offset_voxels=max(abs(value) for value in FULL_OFFSETS),
    )
    offsets_um = tuple(float(value) * VOXEL_UM for value in FULL_OFFSETS)
    native_reports = {
        name: validate_surface_geometry(
            surface,
            volume_shape_zyx=spec.shape_zyx,
            voxel_size_zyx_um=(VOXEL_UM,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for name, sign in (("positive", 1), ("negative", -1))
    }
    defect_names = (
        "discontinuity_edge_count",
        "neighbor_wrap_risk_edge_count",
        "degenerate_quad_count",
        "folded_quad_count",
        "abrupt_normal_flip_edge_count",
        "distorted_quad_count",
    )
    hard_counts = {
        name: int(native_reports["positive"][name]) for name in defect_names
    }
    geometry = {
        "pass": bool(
            all(
                all(int(report[name]) == 0 for name in defect_names)
                and float(report["in_bounds_vertex_fraction"]) == 1.0
                and float(report["normal_valid_vertex_fraction"]) == 1.0
                and float(report["offset_sample_in_bounds_fraction"]) == 1.0
                for report in native_reports.values()
            )
        ),
        "hard_defect_counts": hard_counts,
        "native_reports_by_orientation": native_reports,
        "localized_masks": masks.summary,
        "self_intersection_status": "not evaluated; unknown, never assumed zero",
    }
    # Invalid canvas and hole pixels are intentional mask boundaries.  Whole-mask
    # native geometry is the hard defect gate; every valid one-sided boundary
    # normal is resampled instead of eroding the candidate inward.
    eligible = surface.valid & masks.sample_valid
    support_indices = [FULL_OFFSETS.index(value) for value in SUPPORT_OFFSETS]
    central_indices = [FULL_OFFSETS.index(value) for value in STORED_OFFSETS]
    orientations: dict[str, dict[str, Any]] = {}
    for name, sign in (("positive", 1.0), ("negative", -1.0)):
        frames, validity = sample_offsets(
            sampler,
            surface,
            sign * masks.normals_xyz,
            eligible,
            FULL_OFFSETS,
        )
        requested = np.broadcast_to(eligible, validity.shape)
        sampling_complete = bool(np.all(validity[requested]))
        support = support_summary(
            frames[support_indices], SUPPORT_OFFSETS, eligible
        )
        selected = selected_run_metrics(frames, eligible)
        central_selected = selected_run_metrics(frames[central_indices], eligible)
        centered = eligible & np.asarray(selected["contains_zero"], dtype=bool)
        components, labels = connected_quad_components_summary(
            centered, masks.quad_area_cm2
        )
        largest = components["largest_component"]
        largest_mask = (
            quad_component_vertex_mask(
                labels, int(largest["component_id"]), surface.shape
            )
            if largest is not None
            else np.zeros(surface.shape, dtype=bool)
        )
        no_bridging = bool(
            largest is not None
            and np.array_equal(
                active_quads(largest_mask), labels == largest["component_id"]
            )
        )
        orientations[name] = {
            "frames": frames,
            "validity": validity,
            "sampling_complete": sampling_complete,
            "support": support,
            "selected": selected,
            "central_selected": central_selected,
            "centered": centered,
            "components": components,
            "labels": labels,
            "largest": largest,
            "largest_mask": largest_mask,
            "no_bridging": no_bridging,
            "largest_topology": topology_report(largest_mask)
            if largest is not None
            else None,
            "diagnostics": selected_region_diagnostics(selected, largest_mask),
            "central_diagnostics": selected_region_diagnostics(
                central_selected, largest_mask
            ),
            "area_cm2": float(largest["area_cm2"])
            if largest is not None
            else 0.0,
        }
    positive = orientations["positive"]
    negative = orientations["negative"]
    sign_reversal = {
        "frames_exact_index_reversal": bool(
            np.array_equal(negative["frames"], positive["frames"][::-1])
        ),
        "validity_exact_index_reversal": bool(
            np.array_equal(negative["validity"], positive["validity"][::-1])
        ),
        "largest_masks_identical": bool(
            np.array_equal(negative["largest_mask"], positive["largest_mask"])
        ),
    }
    return {
        "masks": masks,
        "geometry": geometry,
        "eligible": eligible,
        "orientations": orientations,
        "sign_reversal_invariants": sign_reversal,
        "sampling_complete": bool(
            positive["sampling_complete"] and negative["sampling_complete"]
        ),
        "support": positive["support"],
        "selected": positive["selected"],
        "centered": positive["centered"],
        "components": positive["components"],
        "labels": positive["labels"],
        "largest": positive["largest"],
        "largest_mask": positive["largest_mask"],
        "no_bridging": positive["no_bridging"],
        "materialized_topology": topology_report(surface.valid),
        "largest_topology": positive["largest_topology"],
        "diagnostics": positive["diagnostics"],
        "central_diagnostics": positive["central_diagnostics"],
        "area_cm2": positive["area_cm2"],
    }


def calibration_thresholds(preflight: Mapping[str, Any]) -> dict[str, float]:
    control = preflight["published_control_calibration"]
    source = Path(str(control["source_path"]))
    if sha256_file(source) != control["source_sha256"]:
        raise ValueError("published-control calibration source hash changed")
    source_report = json.loads(source.read_text(encoding="utf-8"))
    recomputed = source_report["published_surface_calibration_control"]["recomputed"]
    continuity = recomputed["centered_selected_run_spatial_continuity"]
    clearance = control["competitor_clearance_centered_selected_runs"]["quantiles"]
    return {
        "plus_minus2_support_fraction": float(control["plus_minus2_support_fraction"]),
        "offset0_support_fraction": float(control["offset0_support_fraction"]),
        "continuity_center_jump_p95": float(
            continuity["selected_center_absolute_jump"]["quantiles"]["p95"]
        ),
        "continuity_interval_overlap_fraction": float(
            continuity["interval_overlap_fraction"]
        ),
        "competitor_clearance_minimum": float(clearance["minimum"]),
        "competitor_clearance_p05": float(clearance["p05"]),
        "competitor_clearance_median": float(clearance["median"]),
    }


def orientation_summary(
    orientation: Mapping[str, Any], eligible: NDArray[np.bool_]
) -> dict[str, Any]:
    summary = run_metrics_summary(orientation["selected"], eligible)
    # run_metrics_summary is calibrated to the historical radius-15 constant;
    # explicitly supersede its sentinel for the full radius-46 stack.
    summary["single_run_clearance_sentinel_voxels"] = 46
    summary["single_run_clearance_sentinel_source"] = (
        "explicit radius of the minus46..plus46 stack"
    )
    return {
        "sampling_complete": bool(orientation["sampling_complete"]),
        "plus_minus2_support": orientation["support"],
        "selected_run_metrics_radius46": summary,
        "fresh_centered_components": orientation["components"],
        "fresh_largest_diagnostics_radius46": orientation["diagnostics"],
        "fresh_largest_diagnostics_central_minus15_plus15": orientation[
            "central_diagnostics"
        ],
        "fresh_largest_topology": orientation["largest_topology"],
        "fresh_largest_mask_content_sha256": _array_sha256(
            orientation["largest_mask"]
        ),
        "fresh_largest_area_cm2": float(orientation["area_cm2"]),
    }


def _quantile(summary: Mapping[str, Any], name: str) -> float:
    quantiles = summary.get("quantiles")
    if not isinstance(quantiles, Mapping) or quantiles.get(name) is None:
        return -math.inf
    return float(quantiles[name])


def stored_gate_results(
    post: Mapping[str, Any],
    repair: Mapping[str, Any],
    repaired_mask: NDArray[np.bool_],
    thresholds: Mapping[str, float],
    *,
    minimum_area_cm2: float,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = [
        {
            "name": "repair_strictly_subtractive_idempotent",
            "pass": bool(repair["strict_quad_subset_of_input"] and repair["idempotent"]),
        },
        {"name": "post_repair_native_geometry", "pass": bool(post["geometry"]["pass"])},
        {
            "name": "post_repair_materialized_topology_manifold",
            "pass": not bool(post["materialized_topology"]["nonmanifold_risk"]),
        },
        {
            "name": "both_sign_m7_arrays_are_exact_symmetric_index_reversals",
            "pass": all(post["sign_reversal_invariants"].values()),
        },
    ]
    for sign in ("positive", "negative"):
        orientation = post["orientations"][sign]
        support = orientation["support"]
        # The published control was measured over -15..+15.  Use the central
        # slice for continuity/clearance comparisons while preserving the raw
        # radius-46 diagnostics separately.
        diagnostics = orientation["central_diagnostics"]
        continuity = diagnostics["spatial_continuity"]
        clearance = diagnostics["nearest_competitor_empty_gap"]
        gates.extend(
            [
                {
                    "name": f"{sign}_all_minus46_plus46_m7_samples_valid",
                    "pass": bool(orientation["sampling_complete"]),
                },
                {
                    "name": f"{sign}_plus_minus2_support_matches_published_control",
                    "value": float(support["support_fraction"]),
                    "threshold": float(thresholds["plus_minus2_support_fraction"]),
                    "pass": float(support["support_fraction"])
                    >= float(thresholds["plus_minus2_support_fraction"]),
                },
                {
                    "name": f"{sign}_offset0_support_matches_published_control",
                    "value": float(support["offset0_support_fraction"]),
                    "threshold": float(thresholds["offset0_support_fraction"]),
                    "pass": float(support["offset0_support_fraction"])
                    >= float(thresholds["offset0_support_fraction"]),
                },
                {
                    "name": f"{sign}_fresh_centered_connected_area",
                    "value_cm2": float(orientation["area_cm2"]),
                    "threshold_cm2": float(minimum_area_cm2),
                    "pass": float(orientation["area_cm2"])
                    >= float(minimum_area_cm2),
                },
                {
                    "name": f"{sign}_fresh_component_encodable_without_bridging",
                    "pass": bool(orientation["no_bridging"]),
                },
                {
                    "name": f"{sign}_fresh_component_topology_manifold",
                    "pass": bool(
                        orientation["largest_topology"] is not None
                        and not orientation["largest_topology"]["nonmanifold_risk"]
                    ),
                },
                {
                    "name": f"{sign}_fresh_mask_equals_precommitted_repaired_mask",
                    "pass": bool(
                        np.array_equal(orientation["largest_mask"], repaired_mask)
                    ),
                },
                {
                    "name": f"{sign}_central15_interval_overlap_matches_control",
                    "value": continuity["interval_overlap_fraction"],
                    "threshold": float(
                        thresholds["continuity_interval_overlap_fraction"]
                    ),
                    "pass": bool(
                        continuity["interval_overlap_fraction"] is not None
                        and float(continuity["interval_overlap_fraction"])
                        >= float(thresholds["continuity_interval_overlap_fraction"])
                    ),
                },
                {
                    "name": f"{sign}_central15_center_jump_p95_no_worse_than_control",
                    "value": _quantile(
                        continuity["selected_center_absolute_jump"], "p95"
                    ),
                    "maximum": float(thresholds["continuity_center_jump_p95"]),
                    "pass": _quantile(
                        continuity["selected_center_absolute_jump"], "p95"
                    )
                    <= float(thresholds["continuity_center_jump_p95"]),
                },
                {
                    "name": f"{sign}_central15_competitor_clearance_minimum",
                    "value": _quantile(clearance, "minimum"),
                    "threshold": float(thresholds["competitor_clearance_minimum"]),
                    "pass": _quantile(clearance, "minimum")
                    >= float(thresholds["competitor_clearance_minimum"]),
                },
                {
                    "name": f"{sign}_central15_competitor_clearance_p05",
                    "value": _quantile(clearance, "p05"),
                    "threshold": float(thresholds["competitor_clearance_p05"]),
                    "pass": _quantile(clearance, "p05")
                    >= float(thresholds["competitor_clearance_p05"]),
                },
                {
                    "name": f"{sign}_central15_competitor_clearance_median",
                    "value": _quantile(clearance, "median"),
                    "threshold": float(thresholds["competitor_clearance_median"]),
                    "pass": _quantile(clearance, "median")
                    >= float(thresholds["competitor_clearance_median"]),
                },
            ]
        )
    return gates


def run_stored(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    preflight_path = Path(args.preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "complete":
        raise ValueError("PHerc1203 22-surface preflight is not complete")
    if tuple(preflight["m7"]["shape_zyx"]) != PINNED_VOLUME_SHAPE_ZYX:
        raise ValueError("eligible PHerc1203 m7 volume identity changed")
    candidate_id = str(args.candidate_id)
    segment = one_segment(preflight, candidate_id)
    source_dir = verify_source(segment)
    selected_mask = load_verified_array(
        segment["array_outputs"]["largest_connected_quad_component_vertex_mask"]
    ).astype(bool, copy=False)
    if not np.array_equal(
        active_quads(quad_union_vertex_mask(active_quads(selected_mask))),
        active_quads(selected_mask),
    ):
        raise ValueError("fixed preflight component is not exactly mask encodable")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_cache = output / "m7-metadata-cache"
    m7 = preflight["m7"]
    array_metadata, array_provenance = fetch_cached_json(
        str(m7["array_metadata"]["url"]),
        metadata_cache / "array.zarray.json",
        expected_sha256=str(m7["array_metadata"]["sha256"]),
    )
    attrs, attrs_provenance = fetch_cached_json(
        str(m7["root_attributes"]["url"]),
        metadata_cache / "root.zattrs.json",
        expected_sha256=str(m7["root_attributes"]["sha256"]),
    )
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)
    if spec.shape_zyx != PINNED_VOLUME_SHAPE_ZYX:
        raise ValueError("public m7 .zarray shape differs from eligible scan")
    validate_metadata_axes(attrs, str(m7["array_path"]))
    blosc = Path(args.blosc)
    if not blosc.is_file() or sha256_file(blosc) != m7["codec"]["sha256"]:
        raise ValueError("pinned Blosc codec is missing or changed")
    fetcher = DiskPayloadFetcher(Path(args.m7_cache))
    sampler = PublicZarrChunkSampler(
        root_url=str(m7["root"]),
        array_path=str(m7["array_path"]),
        spec=spec,
        blosc_path=blosc,
        fetcher=fetcher,
    )

    source_asset = load_tifxyz_asset(source_dir, resolution="stored")
    source_masks = derive_geometry_masks(
        source_asset.surface,
        volume_shape_zyx=spec.shape_zyx,
        voxel_um=VOXEL_UM,
        maximum_offset_voxels=max(abs(value) for value in FULL_OFFSETS),
    )
    fixed_dir = output / "materialized-preflight-component"
    fixed_materialized = materialize_masked_tifxyz(
        source_dir,
        fixed_dir,
        selected_mask,
        quad_area_cm2=source_masks.quad_area_cm2,
        role="fixed_preflight_largest_centered_connected_quad_component",
        lineage={
            "preflight_report": str(preflight_path.resolve()),
            "preflight_report_sha256": sha256_file(preflight_path),
            "source_candidate_id": candidate_id,
            "source_component": segment["connected_quad_regions"]["largest_component"],
            "selection_fixed_before_fresh_boundary_normals": True,
        },
        voxel_um=VOXEL_UM,
        default_uuid=candidate_id,
    )
    fixed_asset = load_tifxyz_asset(fixed_dir, resolution="stored")
    pre_repair = evaluate_stored_surface(fixed_asset.surface, sampler, spec)
    if pre_repair["largest"] is None:
        raise RuntimeError("fresh boundary validation produced no positive-area component")
    repaired_mask, repair = conservative_manifold_subset(
        pre_repair["largest_mask"], pre_repair["masks"].quad_area_cm2
    )
    repaired_dir = output / "materialized-topology-repaired-component"
    repaired_materialized = materialize_masked_tifxyz(
        source_dir,
        repaired_dir,
        repaired_mask,
        quad_area_cm2=pre_repair["masks"].quad_area_cm2,
        role="fresh_centered_topology_repaired_component",
        lineage={
            "fixed_preflight_component": str(fixed_dir.resolve()),
            "fresh_pre_repair_mask_content_sha256": _array_sha256(
                pre_repair["largest_mask"]
            ),
            "repair_algorithm": repair["algorithm"],
            "repair_iteration_count": repair["iteration_count"],
            "strictly_subtractive": True,
            "no_coordinate_modification": True,
            "no_per_vertex_retuning": True,
        },
        voxel_um=VOXEL_UM,
        default_uuid=candidate_id,
    )
    repaired_asset = load_tifxyz_asset(repaired_dir, resolution="stored")
    post = evaluate_stored_surface(repaired_asset.surface, sampler, spec)
    thresholds = calibration_thresholds(preflight)
    gates = [
        {
            "name": "fixed_preflight_component_fresh_native_geometry",
            "pass": bool(pre_repair["geometry"]["pass"]),
        },
        {
            "name": "fixed_preflight_component_fresh_m7_sampling_complete",
            "pass": bool(pre_repair["sampling_complete"]),
        },
        {
            "name": "fixed_preflight_component_fresh_area",
            "value_cm2": float(pre_repair["area_cm2"]),
            "threshold_cm2": float(args.minimum_area_cm2),
            "pass": float(pre_repair["area_cm2"])
            >= float(args.minimum_area_cm2),
        },
        *stored_gate_results(
            post,
            repair,
            repaired_mask,
            thresholds,
            minimum_area_cm2=float(args.minimum_area_cm2),
        ),
    ]
    passed = all(bool(item["pass"]) for item in gates)
    arrays = {
        "fixed_preflight_vertex_mask": save_array(
            output / "arrays/fixed-preflight-vertex-mask.npy", selected_mask
        ),
        "fresh_pre_repair_largest_vertex_mask": save_array(
            output / "arrays/fresh-pre-repair-largest-vertex-mask.npy",
            pre_repair["largest_mask"],
        ),
        "topology_repaired_vertex_mask": save_array(
            output / "arrays/topology-repaired-vertex-mask.npy", repaired_mask
        ),
        "post_repair_fresh_normals_xyz": save_array(
            output / "arrays/post-repair-fresh-normals-xyz.npy",
            post["masks"].normals_xyz,
        ),
        "post_repair_positive_m7_offsets_minus46_plus46": save_array(
            output / "arrays/post-repair-positive-m7-offsets-minus46-plus46.npy",
            post["orientations"]["positive"]["frames"],
        ),
        "post_repair_positive_m7_sample_validity": save_array(
            output / "arrays/post-repair-positive-m7-sample-validity.npy",
            post["orientations"]["positive"]["validity"],
        ),
        "post_repair_negative_m7_offsets_minus46_plus46": save_array(
            output / "arrays/post-repair-negative-m7-offsets-minus46-plus46.npy",
            post["orientations"]["negative"]["frames"],
        ),
        "post_repair_negative_m7_sample_validity": save_array(
            output / "arrays/post-repair-negative-m7-sample-validity.npy",
            post["orientations"]["negative"]["validity"],
        ),
        "post_repair_fresh_centered_vertex_mask": save_array(
            output / "arrays/post-repair-fresh-centered-vertex-mask.npy",
            post["centered"],
        ),
        "post_repair_largest_component_vertex_mask": save_array(
            output / "arrays/post-repair-largest-component-vertex-mask.npy",
            post["largest_mask"],
        ),
        "post_repair_quad_component_labels": save_array(
            output / "arrays/post-repair-quad-component-labels.npy", post["labels"]
        ),
        "post_repair_positive_selected_run_center": save_array(
            output / "arrays/post-repair-positive-selected-run-center.npy",
            post["orientations"]["positive"]["selected"][
                "selected_center_offset"
            ],
        ),
        "post_repair_positive_competitor_empty_gap": save_array(
            output / "arrays/post-repair-positive-competitor-empty-gap.npy",
            post["orientations"]["positive"]["selected"][
                "nearest_competitor_empty_gap"
            ],
        ),
        "post_repair_negative_selected_run_center": save_array(
            output / "arrays/post-repair-negative-selected-run-center.npy",
            post["orientations"]["negative"]["selected"][
                "selected_center_offset"
            ],
        ),
        "post_repair_negative_competitor_empty_gap": save_array(
            output / "arrays/post-repair-negative-competitor-empty-gap.npy",
            post["orientations"]["negative"]["selected"][
                "nearest_competitor_empty_gap"
            ],
        ),
    }
    result = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete",
        "candidate_id": candidate_id,
        "verdict": "pass_to_full_resolution_preflight"
        if passed
        else "stop_before_full_resolution",
        "paid_compute_cost_usd": 0.0,
        "raw_ct_network_bytes": 0,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "eligibility_identity": preflight["eligibility_identity"],
        "trust_status": segment["trust_status"],
        "official_source": segment["official_source"],
        "source_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
            "fixed_component": segment["connected_quad_regions"]["largest_component"],
        },
        "source_tifxyz": source_asset.manifest(),
        "materialized_fixed_preflight_component": fixed_materialized,
        "pre_repair_fresh_validation": {
            "native_geometry": compact_stored_geometry(pre_repair["geometry"]),
            "sign_reversal_invariants": pre_repair["sign_reversal_invariants"],
            "by_normal_sign": {
                name: orientation_summary(value, pre_repair["eligible"])
                for name, value in pre_repair["orientations"].items()
            },
        },
        "deterministic_topology_repair": repair,
        "materialized_topology_repaired_component": repaired_materialized,
        "post_repair_fresh_validation": {
            "native_geometry": compact_stored_geometry(post["geometry"]),
            "materialized_topology": post["materialized_topology"],
            "sign_reversal_invariants": post["sign_reversal_invariants"],
            "by_normal_sign": {
                name: orientation_summary(value, post["eligible"])
                for name, value in post["orientations"].items()
            },
            "fresh_centered_components": post["components"],
            "fresh_largest_diagnostics": post["diagnostics"],
            "fresh_largest_topology": post["largest_topology"],
            "fresh_largest_mask_identical_to_repaired_mask": bool(
                np.array_equal(post["largest_mask"], repaired_mask)
            ),
        },
        "published_control_thresholds": thresholds,
        "m7": {
            "root": m7["root"],
            "array_path": m7["array_path"],
            "array_metadata": array_provenance,
            "root_attributes": attrs_provenance,
            "codec_path": str(blosc.resolve()),
            "codec_sha256": sha256_file(blosc),
            "sampler": sampler.manifest(),
            "payload_cache": fetcher.manifest(),
            "stored_offsets_voxels": list(FULL_OFFSETS),
            "normal_signs": ["positive", "negative"],
            "published_control_comparison_slice_voxels": list(STORED_OFFSETS),
        },
        "gate_results": gates,
        "pass_to_full_resolution_preflight": passed,
        "array_outputs": arrays,
        "limitations": [
            "These public auto-grown surfaces are uncurated raw artifacts, not published segments.",
            "Self-intersection remains unknown and is never assumed absent.",
            "Public m7 is a surface-prediction reference, not independent raw-CT proof.",
            "No coordinate was moved, clamped, filled, interpolated, or retuned at stored resolution.",
        ],
    }
    return result, repaired_dir if passed else None


def compact_full_geometry(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
    return {key: report[key] for key in keys}


def full_geometry_pass(report: Mapping[str, Any]) -> bool:
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


def derived_full_shape(
    stored_shape: Sequence[int], scale_yx: Sequence[float]
) -> tuple[int, int]:
    dimensions: list[int] = []
    for size, scale in zip(stored_shape, scale_yx, strict=True):
        ratio = int(size) / float(scale)
        nearest = int(round(ratio))
        tolerance = 1e-6 * max(1.0, abs(ratio))
        dimensions.append(nearest if abs(ratio - nearest) <= tolerance else int(ratio))
    return tuple(dimensions)  # type: ignore[return-value]


def run_full_resolution(
    args: argparse.Namespace,
    stored_validation_path: Path,
    repaired_dir: Path,
) -> dict[str, Any]:
    validation_payload = stored_validation_path.read_bytes()
    validation = json.loads(validation_payload)
    if validation.get("verdict") != "pass_to_full_resolution_preflight":
        raise ValueError("stored validation does not authorize full resampling")
    initial = load_tifxyz_asset(
        repaired_dir,
        resolution="full",
        interpolation="linear",
        maximum_surface_pixels=int(args.maximum_full_pixels),
    )
    expected_shape = derived_full_shape(initial.stored_shape, initial.scale_yx)
    initial_mask = initial.surface.valid
    initial_quads = scalable_active_quads(initial_mask)
    render_mask = scalable_quad_union_vertex_mask(initial_quads)
    if np.any(render_mask & ~initial_mask):
        raise RuntimeError("full-raster orphan cleanup added a mask pixel")
    if not np.array_equal(scalable_active_quads(render_mask), initial_quads):
        raise RuntimeError("full-raster orphan cleanup changed the quad set")
    surface = SurfaceGrid.from_tifxyz(
        initial.surface.x,
        initial.surface.y,
        initial.surface.z,
        mask=render_mask,
    )
    asset = replace(
        initial,
        surface=surface,
        mask_source=initial.mask_source
        + "; strictly-subtractive-full-resolution-unused-vertex-cleanup",
    )
    topology = scalable_topology_report(surface.valid)
    area = scalable_physical_area_cm2(surface, voxel_um=VOXEL_UM)
    offsets_um = tuple(float(value) * VOXEL_UM for value in FULL_OFFSETS)
    reports: dict[str, Any] = {}
    for name, sign in (("positive", 1), ("negative", -1)):
        raw = validate_surface_geometry(
            surface,
            volume_shape_zyx=PINNED_VOLUME_SHAPE_ZYX,
            voxel_size_zyx_um=(VOXEL_UM,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        reports[name] = {**compact_full_geometry(raw), "pass": full_geometry_pass(raw)}
    gates = [
        {
            "name": "full_shape_is_metadata_scale_derived",
            "value": list(asset.output_shape),
            "expected": list(expected_shape),
            "pass": tuple(asset.output_shape) == expected_shape,
        },
        {
            "name": "full_orphan_cleanup_strictly_subtractive_quad_preserving",
            "removed_unused_vertex_count": int(np.count_nonzero(initial_mask & ~render_mask)),
            "removed_quad_count": 0,
            "pass": bool(
                np.all(~render_mask | initial_mask)
                and np.array_equal(scalable_active_quads(render_mask), initial_quads)
            ),
        },
        {
            "name": "full_induced_quad_closure_exact",
            "pass": bool(topology["induced_quad_closure_exact"]),
        },
        {
            "name": "full_single_manifold_quad_component",
            "pass": not bool(topology["nonmanifold_risk"]),
        },
        {
            "name": "full_physical_area",
            "value_cm2": area,
            "threshold_cm2": float(args.minimum_area_cm2),
            "pass": area >= float(args.minimum_area_cm2),
        },
        {"name": "full_positive_native_qc_minus46_plus46", "pass": reports["positive"]["pass"]},
        {"name": "full_negative_native_qc_minus46_plus46", "pass": reports["negative"]["pass"]},
    ]
    passed = all(bool(item["pass"]) for item in gates)
    output = Path(args.output) / "full-resolution-preflight"
    output.mkdir(parents=True, exist_ok=True)
    mask_path = output / "full-resolution-surface-valid-mask.tif"
    tifffile.imwrite(
        mask_path,
        surface.valid.astype(np.uint8) * 255,
        photometric="minisblack",
        compression="deflate",
    )
    return {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "complete",
        "candidate_id": str(args.candidate_id),
        "verdict": "pass_to_raw_ct_planning" if passed else "stop_before_raw_ct",
        "paid_compute_cost_usd": 0.0,
        "raw_ct_network_bytes": 0,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "source_stored_validation": {
            "path": str(stored_validation_path.resolve()),
            "sha256": sha256_bytes(validation_payload),
        },
        "full_resolution_tifxyz": asset.manifest(),
        "surface_interpolation": {
            "method": "canonical mask-aware linear TIFFXYZ resampling",
            "stored_shape": list(asset.stored_shape),
            "full_shape": list(asset.output_shape),
            "scale_yx": list(asset.scale_yx),
            "expected_full_shape": list(expected_shape),
        },
        "strictly_subtractive_orphan_cleanup": {
            "initial_valid_vertex_count": int(np.count_nonzero(initial_mask)),
            "initial_mask_content_sha256": _array_sha256(initial_mask),
            "removed_unused_vertex_count": int(np.count_nonzero(initial_mask & ~render_mask)),
            "added_vertex_count": int(np.count_nonzero(render_mask & ~initial_mask)),
            "initial_quad_count": int(np.count_nonzero(initial_quads)),
            "final_quad_count": int(np.count_nonzero(scalable_active_quads(render_mask))),
        },
        "explicit_full_resolution_valid_mask": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "array_content_sha256": _array_sha256(surface.valid),
            "valid_vertex_count": int(np.count_nonzero(surface.valid)),
        },
        "topology": topology,
        "physical_area_cm2": area,
        "normal_offset_voxels": list(FULL_OFFSETS),
        "normal_offset_micrometers": list(offsets_um),
        "native_geometry_by_sign": reports,
        "gate_results": gates,
        "pass_to_raw_ct_planning": passed,
        "limitations": [
            "Self-intersection remains unknown and is never assumed absent.",
            "Combinatorial manifold topology does not prove geometric non-self-intersection.",
            "No raw CT was fetched; this is the last free surface gate only.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--m7-cache", type=Path, default=DEFAULT_M7_CACHE)
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--minimum-area-cm2", type=float, default=MINIMUM_AREA_CM2)
    parser.add_argument("--maximum-full-pixels", type=int, default=10_000_000)
    parser.add_argument("--stored-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output = Path(args.output_root) / str(args.candidate_id)
    stored_path = Path(args.output) / "stored-validation.json"
    full_path = Path(args.output) / "full-resolution-preflight.json"
    try:
        stored, repaired_dir = run_stored(args)
        _atomic_json(stored_path, stored)
        if repaired_dir is not None and not args.stored_only:
            full = run_full_resolution(args, stored_path, repaired_dir)
            _atomic_json(full_path, full)
        else:
            full = None
        manifest = write_manifest(Path(args.output))
    except Exception as error:
        failure = {
            "schema_version": 1,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "error",
            "candidate_id": str(args.candidate_id),
            "error_type": type(error).__name__,
            "error": str(error),
            "paid_compute_cost_usd": 0.0,
            "raw_ct_network_bytes": 0,
        }
        attempt = Path(args.output) / f"attempt-error-{int(time.time())}.json"
        _atomic_json(attempt, failure)
        print(json.dumps({**failure, "attempt_evidence": str(attempt)}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "candidate_id": str(args.candidate_id),
                "stored_verdict": stored["verdict"],
                "stored_area_cm2": stored["post_repair_fresh_validation"]
                ["fresh_centered_components"]["largest_component"]["area_cm2"],
                "full_verdict": full["verdict"] if full is not None else None,
                "stored_report": str(stored_path.resolve()),
                "full_report": str(full_path.resolve()) if full is not None else None,
                "manifest": manifest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
