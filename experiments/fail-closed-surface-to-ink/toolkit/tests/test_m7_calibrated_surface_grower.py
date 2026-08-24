from __future__ import annotations

import copy
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from m7_calibrated_surface_grower import (  # noqa: E402
    build_parser,
    strict_selected_run_gates,
)


def passing_calibrated(vertex_count: int = 25):
    report = {
        "folded_quad_count": 0,
        "abrupt_normal_flip_edge_count": 0,
        "degenerate_quad_count": 0,
    }
    return {
        "geometry_pass": True,
        "sampling_complete": True,
        "support": {
            "valid_vertex_count": vertex_count,
            "supported_vertex_count": vertex_count,
            "offset0_supported_vertex_count": vertex_count,
        },
        "native_reports_by_orientation": {
            "positive": dict(report),
            "negative": dict(report),
        },
        "selected_run_records": [
            {
                "row": index // 5,
                "column": index % 5,
                "sample_valid": True,
                "pass": True,
                "selected_run_center_offset": 0.0,
                "nearest_competitor_empty_gap_voxels": 2 if index % 2 else None,
            }
            for index in range(vertex_count)
        ],
    }


def gate_by_name(result, name: str):
    return next(gate for gate in result["gates"] if gate["name"] == name)


def test_strict_selected_run_gate_passes_exact_5x5() -> None:
    result = strict_selected_run_gates(
        passing_calibrated(), accepted_radius=2, target_radius=2
    )
    assert result["pass"]
    assert result["required_vertex_count"] == 25
    assert result["centered_selected_run_pass_count"] == 25


def test_strict_selected_run_gate_rejects_off_center_and_zero_gap() -> None:
    calibrated = passing_calibrated()
    calibrated["selected_run_records"][3]["selected_run_center_offset"] = 2.5
    calibrated["selected_run_records"][5][
        "nearest_competitor_empty_gap_voxels"
    ] = 0
    result = strict_selected_run_gates(
        calibrated, accepted_radius=2, target_radius=2
    )
    assert not result["pass"]
    assert not gate_by_name(
        result, "exhaustive_centered_selected_run_continuity"
    )["pass"]
    assert not gate_by_name(
        result, "minimum_competitor_empty_gap_one_voxel"
    )["pass"]


def test_strict_selected_run_gate_requires_both_signs_and_zero_folds() -> None:
    calibrated = passing_calibrated()
    calibrated["native_reports_by_orientation"]["negative"][
        "folded_quad_count"
    ] = 1
    result = strict_selected_run_gates(
        calibrated, accepted_radius=2, target_radius=2
    )
    assert not result["pass"]
    assert not gate_by_name(result, "zero_folds_flips_or_degenerate_quads")[
        "pass"
    ]

    missing_sign = copy.deepcopy(passing_calibrated())
    del missing_sign["native_reports_by_orientation"]["negative"]
    result = strict_selected_run_gates(
        missing_sign, accepted_radius=2, target_radius=2
    )
    assert not gate_by_name(result, "native_geometry_both_orientations")["pass"]


def test_parser_accepts_explicit_pherc1203_identity() -> None:
    args = build_parser().parse_args(
        [
            "--seed-xyz",
            "3400",
            "3490",
            "12960",
            "--volume-shape-zyx",
            "18977",
            "6844",
            "6844",
            "--voxel-um",
            "9.362",
            "--target-volume",
            "PHerc1203-eligible-9.362um",
            "--output",
            "out",
        ]
    )
    assert tuple(args.volume_shape_zyx) == (18977, 6844, 6844)
    assert args.target_volume == "PHerc1203-eligible-9.362um"
    assert args.minimum_competitor_empty_gap_voxels == 1
