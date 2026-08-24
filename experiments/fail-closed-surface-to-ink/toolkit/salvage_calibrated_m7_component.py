#!/usr/bin/env python3
"""Extract and freshly validate a deterministic calibrated-m7 surface component.

This is a strictly subtractive salvage: coordinates never move.  A source-normal
mask is precommitted from saved public-m7 frames, then mask-boundary normals are
recomputed and the same rule is iterated to a fixed point.  Other foreground
runs are calibrated diagnostics, not a veto; the selected run must cross zero
and have midpoint within two voxels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from audit_pherc0800_published_segments import (
    DiskPayloadFetcher,
    connected_quad_components_summary,
    quad_component_vertex_mask,
    selected_region_diagnostics,
    selected_run_metrics,
)
from preflight_debug_m7 import (
    DEFAULT_TRANSECT_OFFSETS,
    derive_geometry_masks,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import ZarrV2ArraySpec, fetch_json, sha256_file
from sweep_public_m7_uniform_offsets import hard_geometry_summary
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_pherc0800_masked_candidate import (
    active_quads,
    conservative_manifold_subset,
    materialize_masked_tifxyz,
    physical_area_from_mask,
    topology_report,
)
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    _array_sha256,
    _atomic_json,
    validate_metadata_axes,
)


def calibrated_vertex_mask(
    metrics: Mapping[str, np.ndarray],
    eligible: np.ndarray,
    *,
    maximum_center_offset: float,
) -> np.ndarray:
    center = np.asarray(metrics["selected_center_offset"], dtype=np.float64)
    return (
        np.asarray(eligible, dtype=bool)
        & np.asarray(metrics["contains_zero"], dtype=bool)
        & np.isfinite(center)
        & (np.abs(center) <= float(maximum_center_offset))
    )


def largest_quad_mask(
    vertex_mask: np.ndarray, quad_area_cm2: np.ndarray
) -> tuple[np.ndarray, Mapping[str, Any]]:
    components, labels = connected_quad_components_summary(
        np.asarray(vertex_mask, dtype=bool), quad_area_cm2
    )
    largest = components["largest_component"]
    if largest is None:
        raise ValueError("calibrated mask contains no positive-area quad component")
    mask = quad_component_vertex_mask(
        labels, int(largest["component_id"]), vertex_mask.shape
    )
    if not np.array_equal(active_quads(mask), labels == int(largest["component_id"])):
        raise RuntimeError("largest component cannot be encoded without bridging")
    return mask, {"components": components, "largest": largest}


def evaluate(
    source: Path,
    mask: np.ndarray,
    sampler: PublicZarrChunkSampler,
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
    maximum_center_offset: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    asset = load_tifxyz_asset(source, resolution="stored")
    if not np.array_equal(asset.surface.valid, mask):
        raise ValueError("reloaded materialized mask differs from expected mask")
    geometry_masks, geometry = hard_geometry_summary(
        asset.surface,
        volume_shape_zyx=volume_shape_zyx,
        voxel_um=voxel_um,
    )
    eligible = asset.surface.valid & geometry_masks.sample_valid
    frames, sample_validity = sample_offsets(
        sampler,
        asset.surface,
        geometry_masks.normals_xyz,
        eligible,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(eligible, sample_validity.shape)
    sampling_complete = bool(np.all(sample_validity[requested]))
    metrics = selected_run_metrics(frames, eligible)
    centered = calibrated_vertex_mask(
        metrics, eligible, maximum_center_offset=maximum_center_offset
    )
    candidate, component = largest_quad_mask(centered, geometry_masks.quad_area_cm2)
    repaired, repair = conservative_manifold_subset(
        candidate, geometry_masks.quad_area_cm2
    )
    diagnostics = selected_region_diagnostics(metrics, repaired)
    area = physical_area_from_mask(repaired, geometry_masks.quad_area_cm2)
    topology = topology_report(repaired)
    competitor = np.asarray(metrics["nearest_competitor_empty_gap"], dtype=np.float64)
    selected = repaired & np.asarray(metrics["selected_valid"], dtype=bool)
    finite_gaps = competitor[selected & np.isfinite(competitor)]
    record = {
        "source": str(source.resolve()),
        "input_mask_content_sha256": _array_sha256(mask.astype(bool)),
        "input_valid_vertex_count": int(np.count_nonzero(mask)),
        "input_area_cm2": physical_area_from_mask(mask, geometry_masks.quad_area_cm2),
        "native_geometry_pass": bool(geometry["pass"]),
        "native_hard_defect_counts": geometry["hard_defect_counts"],
        "sampling_complete": sampling_complete,
        "eligible_vertex_count": int(np.count_nonzero(eligible)),
        "calibrated_vertex_count_before_component": int(np.count_nonzero(centered)),
        "largest_component": component,
        "topology_repair": repair,
        "output_mask_content_sha256": _array_sha256(repaired.astype(bool)),
        "output_valid_vertex_count": int(np.count_nonzero(repaired)),
        "output_active_quad_count": int(np.count_nonzero(active_quads(repaired))),
        "output_area_cm2": area,
        "topology": topology,
        "selected_run_diagnostics": diagnostics,
        "competitor_gap_min": int(np.min(finite_gaps)) if finite_gaps.size else None,
        "frames_content_sha256": _array_sha256(frames),
        "sample_validity_content_sha256": _array_sha256(sample_validity),
        "passes_iteration_gates": bool(
            geometry["pass"]
            and sampling_complete
            and not topology["nonmanifold_risk"]
            and diagnostics["selected_run_vertex_count"]
            == int(np.count_nonzero(repaired))
            and diagnostics["centered_selected_run_vertex_count"]
            == int(np.count_nonzero(repaired))
            and diagnostics["selected_run_center_absolute"]["quantiles"]["maximum"]
            <= maximum_center_offset
            and (finite_gaps.size == 0 or np.min(finite_gaps) >= 1)
        ),
    }
    return repaired, record


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    source_asset = load_tifxyz_asset(source, resolution="stored")
    source_frames_path = Path(args.source_frames)
    frames = np.load(source_frames_path, allow_pickle=False)
    if frames.shape != (31, *source_asset.surface.shape):
        raise ValueError("source frame shape differs from source TIFFXYZ")

    source_masks = derive_geometry_masks(
        source_asset.surface,
        volume_shape_zyx=args.volume_shape_zyx,
        voxel_um=args.voxel_um,
    )
    source_metrics = selected_run_metrics(frames, source_masks.sample_valid)
    source_candidates = calibrated_vertex_mask(
        source_metrics,
        ~source_masks.forbidden & source_masks.sample_valid,
        maximum_center_offset=args.maximum_center_offset,
    )
    initial, component = largest_quad_mask(
        source_candidates, source_masks.quad_area_cm2
    )
    initial, initial_repair = conservative_manifold_subset(
        initial, source_masks.quad_area_cm2
    )
    initial_area = physical_area_from_mask(initial, source_masks.quad_area_cm2)
    if initial_area < args.minimum_area_cm2:
        raise ValueError(
            f"precommitted component {initial_area:.9f} cm2 is below minimum"
        )

    array_metadata, array_provenance = fetch_json(
        args.m7_root.rstrip("/") + "/" + args.m7_array_path.strip("/") + "/.zarray"
    )
    attributes, attrs_provenance = fetch_json(args.m7_root.rstrip("/") + "/.zattrs")
    validate_metadata_axes(attributes, args.m7_array_path)
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)
    if tuple(spec.shape_zyx) != tuple(args.volume_shape_zyx):
        raise ValueError("m7 shape differs from declared volume shape")
    sampler = PublicZarrChunkSampler(
        root_url=args.m7_root,
        array_path=args.m7_array_path,
        spec=spec,
        blosc_path=Path(args.blosc),
        fetcher=DiskPayloadFetcher(Path(args.m7_cache)),
    )

    stages: list[dict[str, Any]] = []
    current_mask = initial
    for iteration in range(1, args.maximum_iterations + 1):
        stage_dir = output / f"stage-{iteration:02d}-tifxyz"
        materialized = materialize_masked_tifxyz(
            source,
            stage_dir,
            current_mask,
            quad_area_cm2=source_masks.quad_area_cm2,
            role=f"calibrated_m7_fixed_point_stage_{iteration:02d}",
            lineage={
                "source_frames": str(source_frames_path.resolve()),
                "source_frames_sha256": sha256_file(source_frames_path),
                "selection": (
                    "largest 4-neighbor quad component outside one-cell source "
                    "geometry defects whose nearest selected public-m7 run crosses "
                    "offset0 and has absolute midpoint <=2 voxels"
                ),
                "iteration": iteration,
                "strictly_subtractive": True,
            },
            voxel_um=args.voxel_um,
            default_uuid=args.candidate_id,
        )
        fresh_mask, evaluation = evaluate(
            stage_dir,
            current_mask,
            sampler,
            volume_shape_zyx=args.volume_shape_zyx,
            voxel_um=args.voxel_um,
            maximum_center_offset=args.maximum_center_offset,
        )
        stable = bool(np.array_equal(fresh_mask, current_mask))
        stages.append(
            {
                "iteration": iteration,
                "materialized": materialized,
                "evaluation": evaluation,
                "fixed_point": stable,
            }
        )
        if not evaluation["passes_iteration_gates"]:
            break
        if evaluation["output_area_cm2"] < args.minimum_area_cm2:
            break
        if np.any(fresh_mask & ~current_mask):
            raise RuntimeError("fresh-boundary iteration added a vertex")
        if stable:
            current_mask = fresh_mask
            break
        current_mask = fresh_mask

    final = stages[-1]
    fixed_point = bool(final["fixed_point"])
    area = float(final["evaluation"]["output_area_cm2"])
    passed = bool(
        fixed_point
        and final["evaluation"]["passes_iteration_gates"]
        and area >= args.minimum_area_cm2
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": args.candidate_id,
        "verdict": "pass_stored_calibrated_geometry" if passed else "no_go",
        "go": passed,
        "paid_compute_cost_usd": 0.0,
        "policy": {
            "coordinate_changes": "none; masks only; every iteration subtractive",
            "selected_run": "nearest public-m7 run must cross offset0",
            "maximum_selected_run_midpoint_absolute_voxels": args.maximum_center_offset,
            "other_runs": "diagnostic; minimum empty gap >=1 voxel",
            "minimum_area_cm2": args.minimum_area_cm2,
            "self_intersection": "not yet evaluated; required before raw render",
        },
        "inputs": {
            "source": str(source.resolve()),
            "source_frames": str(source_frames_path.resolve()),
            "source_frames_sha256": sha256_file(source_frames_path),
            "m7_root": args.m7_root,
            "m7_array_path": args.m7_array_path,
            "m7_array_metadata_provenance": array_provenance,
            "m7_attributes_provenance": attrs_provenance,
            "voxel_um": args.voxel_um,
            "volume_shape_zyx": list(args.volume_shape_zyx),
        },
        "precommit": {
            "source_geometry": source_masks.summary,
            "source_candidate_vertex_count": int(np.count_nonzero(source_candidates)),
            "largest_component": component,
            "initial_topology_repair": initial_repair,
            "initial_mask_content_sha256": _array_sha256(initial.astype(bool)),
            "initial_area_cm2": initial_area,
        },
        "iterations": stages,
        "final": {
            "fixed_point": fixed_point,
            "area_cm2": area,
            "valid_vertex_count": final["evaluation"]["output_valid_vertex_count"],
            "active_quad_count": final["evaluation"]["output_active_quad_count"],
            "tifxyz": final["materialized"]["directory"],
            "self_intersection_status": "unknown_not_computed",
        },
    }
    _atomic_json(output / "validation.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate-id", required=True)
    value.add_argument("--source", type=Path, required=True)
    value.add_argument("--source-frames", type=Path, required=True)
    value.add_argument("--m7-root", required=True)
    value.add_argument("--m7-array-path", default="0")
    value.add_argument("--m7-cache", type=Path, required=True)
    value.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--voxel-um", type=float, required=True)
    value.add_argument("--volume-shape-zyx", type=int, nargs=3, required=True)
    value.add_argument("--maximum-center-offset", type=float, default=2.0)
    value.add_argument("--minimum-area-cm2", type=float, default=0.5)
    value.add_argument("--maximum-iterations", type=int, default=4)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "error",
            "candidate_id": str(args.candidate_id),
            "verdict": "validation_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "paid_compute_cost_usd": 0.0,
        }
        Path(args.output).mkdir(parents=True, exist_ok=True)
        _atomic_json(Path(args.output) / "validation.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "go": result["go"],
                "area_cm2": result["final"]["area_cm2"],
                "output": str(Path(args.output) / "validation.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if result["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
