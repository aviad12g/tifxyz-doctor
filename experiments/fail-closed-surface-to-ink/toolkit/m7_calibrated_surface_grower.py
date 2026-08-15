#!/usr/bin/env python3
"""Fail-closed complete-ring bootstrap using calibrated selected m7 runs.

This is the bounded counterpart to ``m7_calibrated_directional_grower``.  It
uses the same empirically calibrated projection semantics: the unique run
crossing offset 0 is selected and recentered, while other, separated runs are
retained as diagnostics rather than treated as an automatic sheet-switch.
All proposal distance, PCA, normal, spacing, geometry, ambiguity-cluster, and
bounds constraints remain those of the locked complete-ring grower.

The strict bootstrap gate additionally requires every requested vertex to be
supported within +/-2 and at offset 0, every native-normal transect to have a
centered selected run, competitor clearance of at least one empty voxel, and
zero folds or abrupt flips in both normal orientations.  It uses no raw CT,
ink inference, model execution, paid GPU, or RunPod.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from m7_calibrated_directional_grower import (
    CalibrationGate,
    DEFAULT_CALIBRATION,
    project_target_calibrated,
    surface_calibrated_gate,
)
from m7_locked_surface_grower import (
    ALGORITHM_VERSION as LOCKED_ALGORITHM_VERSION,
    GrowConfig,
    GrowResult,
    PublicM7BinaryAccessor,
    _atomic_json,
    _write_artifact,
    grow_complete_rings,
    load_public_sampler,
    triangulated_area_cm2,
)
from preflight_debug_m7 import DEFAULT_VOLUME_SHAPE_ZYX, DEFAULT_VOXEL_UM
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from validate_public_m7_patch import DEFAULT_M7_ROOT, PublicZarrChunkSampler


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-calibrated-complete-rings-v1.0"


def strict_selected_run_gates(
    calibrated: Mapping[str, Any],
    *,
    accepted_radius: int,
    target_radius: int,
) -> Mapping[str, Any]:
    """Apply the precommitted all-vertex selected-run bootstrap thresholds."""

    required_vertices = (2 * int(target_radius) + 1) ** 2
    support = calibrated["support"]
    records = list(calibrated.get("selected_run_records", ()))
    reports = calibrated["native_reports_by_orientation"]
    both_orientations = set(reports) == {"positive", "negative"}
    no_folds_or_flips = bool(
        both_orientations
        and all(
            int(report["folded_quad_count"]) == 0
            and int(report["abrupt_normal_flip_edge_count"]) == 0
            and int(report["degenerate_quad_count"]) == 0
            for report in reports.values()
        )
    )
    selected_centered = [
        record
        for record in records
        if bool(record.get("sample_valid", False))
        and bool(record.get("pass", False))
        and abs(float(record.get("selected_run_center_offset", math.inf))) <= 2.0
    ]
    competitor_records = [
        record
        for record in records
        if record.get("nearest_competitor_empty_gap_voxels") is not None
    ]
    competitor_clearance_pass = all(
        int(record["nearest_competitor_empty_gap_voxels"]) >= 1
        for record in competitor_records
    )
    gates = [
        {
            "name": "complete_target_radius",
            "pass": int(accepted_radius) >= int(target_radius),
            "value": int(accepted_radius),
            "threshold": int(target_radius),
        },
        {
            "name": "exact_complete_vertex_count",
            "pass": int(support["valid_vertex_count"]) == required_vertices,
            "value": int(support["valid_vertex_count"]),
            "threshold": required_vertices,
        },
        {
            "name": "native_geometry_both_orientations",
            "pass": bool(calibrated["geometry_pass"] and both_orientations),
        },
        {"name": "zero_folds_flips_or_degenerate_quads", "pass": no_folds_or_flips},
        {
            "name": "all_requested_m7_samples_valid",
            "pass": bool(calibrated["sampling_complete"]),
        },
        {
            "name": "all_vertices_supported_within_plus_minus_2",
            "pass": int(support["supported_vertex_count"]) == required_vertices,
            "value": int(support["supported_vertex_count"]),
            "threshold": required_vertices,
        },
        {
            "name": "all_vertices_supported_at_offset0",
            "pass": int(support["offset0_supported_vertex_count"]) == required_vertices,
            "value": int(support["offset0_supported_vertex_count"]),
            "threshold": required_vertices,
        },
        {
            "name": "exhaustive_centered_selected_run_continuity",
            "pass": len(records) == required_vertices
            and len(selected_centered) == required_vertices,
            "value": len(selected_centered),
            "threshold": required_vertices,
        },
        {
            "name": "minimum_competitor_empty_gap_one_voxel",
            "pass": competitor_clearance_pass,
            "value": min(
                (
                    int(record["nearest_competitor_empty_gap_voxels"])
                    for record in competitor_records
                ),
                default=None,
            ),
            "threshold": 1,
            "competitor_vertex_count": len(competitor_records),
        },
    ]
    return {
        "pass": all(bool(gate["pass"]) for gate in gates),
        "gates": gates,
        "required_vertex_count": required_vertices,
        "centered_selected_run_pass_count": len(selected_centered),
        "competitor_vertex_count": len(competitor_records),
    }


def validate_bootstrap(
    result: GrowResult,
    sampler: PublicZarrChunkSampler,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> Mapping[str, Any]:
    calibrated = surface_calibrated_gate(
        result.points_zyx, sampler, gate, voxel_um=voxel_um
    )
    strict = strict_selected_run_gates(
        calibrated,
        accepted_radius=result.accepted_radius,
        target_radius=result.target_radius,
    )
    return {
        **calibrated,
        "pass": strict["pass"],
        "gates": strict["gates"],
        "required_vertex_count": strict["required_vertex_count"],
        "centered_selected_run_pass_count": strict[
            "centered_selected_run_pass_count"
        ],
        "competitor_vertex_count": strict["competitor_vertex_count"],
        "accepted_radius": result.accepted_radius,
        "target_radius": result.target_radius,
        "stored_shape": list(result.points_zyx.shape[:2]),
        "area_cm2": triangulated_area_cm2(result.points_zyx, voxel_um),
        "run_count_gate": (
            "the unique offset0-crossing run is authoritative; distant runs are "
            "recorded diagnostics; minimum empty gap is one voxel"
        ),
        "self_intersection_status": "not localized; unknown; never assumed zero",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-xyz", type=float, nargs=3, required=True)
    parser.add_argument("--target-radius", type=int, default=2)
    parser.add_argument("--minimum-output-radius", type=int, default=2)
    parser.add_argument("--spacing-voxels", type=float, default=20.0)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--target-volume", default=Path(DEFAULT_M7_ROOT).name
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--minimum-competitor-empty-gap-voxels", type=int, default=1
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    output = Path(args.output)
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        if int(args.minimum_competitor_empty_gap_voxels) != 1:
            raise ValueError(
                "this precommitted bootstrap requires exactly one empty competitor-gap voxel"
            )
        calibration_path = Path(args.calibration)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        conclusion = calibration["conclusion"]
        if conclusion.get("exactly_one_run_is_valid_identity_gate") is not False:
            raise ValueError("calibration does not falsify the exactly-one-run identity gate")
        gate = CalibrationGate(
            minimum_support_fraction_pm2=1.0,
            minimum_offset0_fraction=1.0,
            minimum_competitor_empty_gap_voxels=1,
        )
        config = GrowConfig(
            target_radius=int(args.target_radius),
            minimum_output_radius=int(args.minimum_output_radius),
            spacing_voxels=float(args.spacing_voxels),
        )
        sampler, m7_input = load_public_sampler(
            str(args.m7_root),
            str(args.m7_array_path),
            Path(args.blosc),
            args.volume_shape_zyx,
        )
        accessor = PublicM7BinaryAccessor(sampler)

        def calibrated_projector(accessor_arg, target, prior, grow_config, *, inward_neighbor_count):
            return project_target_calibrated(
                accessor_arg,
                target,
                prior,
                grow_config,
                gate,
                inward_neighbor_count=inward_neighbor_count,
            )

        result = grow_complete_rings(
            accessor,
            args.seed_xyz,
            config,
            voxel_um=float(args.voxel_um),
            projector=calibrated_projector,
        )
        validation = validate_bootstrap(
            result, sampler, gate, voxel_um=float(args.voxel_um)
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "parent_locked_algorithm_version": LOCKED_ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_calibrated_bootstrap" if validation["pass"] else "no_go",
            "verdict_scope": (
                "bounded public-m7 selected-run geometry only; no raw or ink inference"
            ),
            "paid_compute_cost_usd": 0.0,
            "elapsed_seconds": time.monotonic() - started,
            "inputs": {
                "m7": m7_input,
                "seed_xyz": [float(value) for value in args.seed_xyz],
                "expected_volume_shape_zyx": [
                    int(value) for value in args.volume_shape_zyx
                ],
                "target_volume": str(args.target_volume),
                "calibration": {
                    "path": str(calibration_path.resolve()),
                    "sha256": sha256_file(calibration_path),
                    "source_validation_sha256": calibration["source_validation"][
                        "original_validation_sha256"
                    ],
                    "conclusion": conclusion,
                },
            },
            "projection_semantics": {
                "selected_run": "unique run crossing offset0",
                "competing_runs": "recorded diagnostic, not standalone veto",
                "minimum_competitor_empty_gap_voxels": 1,
                "unchanged_locked_constraints": [
                    "proposal distance",
                    "PCA planarity",
                    "normal coherence",
                    "spacing",
                    "projection-cluster ambiguity",
                    "native geometry",
                    "bounds",
                ],
            },
            "config": asdict(config),
            "strict_gate": asdict(gate),
            "growth": {
                "accepted_radius": result.accepted_radius,
                "target_radius": result.target_radius,
                "stored_shape": list(result.points_zyx.shape[:2]),
                "passed_minimum_size": result.passed_minimum_size,
                "stop_reason": result.stop_reason,
                "seed": result.seed_record,
                "rings": list(result.ring_records),
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "Distant +/-15 competing runs are diagnostics, not an identity veto.",
                "Self-intersection is not localized and is never assumed zero.",
                "No raw CT, ink inference, detector, paid GPU, or RunPod was used.",
            ],
        }
        _write_artifact(
            output,
            result,
            validation,
            provenance,
            seed_xyz=args.seed_xyz,
            config=config,
            voxel_um=float(args.voxel_um),
            target_volume=str(args.target_volume),
            algorithm_version=ALGORITHM_VERSION,
        )
    except Exception as error:
        error_path = output.with_name(output.name + ".error.json")
        if not error_path.exists():
            _atomic_json(
                error_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "algorithm_version": ALGORITHM_VERSION,
                    "status": "error",
                    "verdict": "no_go",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "output_requested": str(output),
                    "paid_compute_cost_usd": 0.0,
                },
            )
        print(f"FAIL CLOSED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted_radius": result.accepted_radius,
                "shape": list(result.points_zyx.shape[:2]),
                "validation_pass": validation["pass"],
                "stop_reason": result.stop_reason,
                "area_cm2": validation["area_cm2"],
            },
            sort_keys=True,
        )
    )
    return 0 if validation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
