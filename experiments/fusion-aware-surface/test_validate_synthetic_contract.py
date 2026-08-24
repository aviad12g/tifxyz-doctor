from __future__ import annotations

import copy

import pytest
import validate_final_results as validator


def rates(*, neighbours: int, detected: int, fused: int, controls: int, splits: int) -> dict:
    return {
        "neighbour_sites": neighbours,
        "detected_neighbour_sites": detected,
        "fused_detected_sites": fused,
        "control_sites": controls,
        "false_split_sites": splits,
        "site_center_detection_rate": detected / max(neighbours, 1),
        "conditional_fusion_rate": fused / max(detected, 1),
        "false_split_rate": splits / max(controls, 1),
    }


def run_record(run: str) -> dict:
    cells = validator.synthetic_cells()
    arm, _, seed_text = run.partition("_seed")
    seed = int(seed_text) if seed_text else 0
    fused_by_seed = {11: 35, 23: 39, 47: 42}
    rows = {}
    for name, kind in cells.items():
        if kind == "primary":
            if arm == "gap8":
                rows[name] = rates(
                    neighbours=100,
                    detected=99,
                    fused=fused_by_seed[seed],
                    controls=0,
                    splits=0,
                )
            else:
                rows[name] = rates(
                    neighbours=100,
                    detected=100,
                    fused=50,
                    controls=0,
                    splits=0,
                )
        else:
            rows[name] = rates(
                neighbours=0,
                detected=0,
                fused=0,
                controls=100,
                splits=6 if arm == "gap8" else 5,
            )
    primary = validator.pool(
        [rows[name] for name, kind in cells.items() if kind == "primary"],
        f"fixture.{run}.primary",
    )
    controls = validator.pool(
        [rows[name] for name, kind in cells.items() if kind == "single_sheet_control"],
        f"fixture.{run}.control",
    )
    threshold_record = {
        "primary_pooled": primary,
        "single_sheet_control_pooled": controls,
        "per_cell": rows,
    }
    return {
        "selected_threshold": 0.5,
        "thresholds": {
            "0.4": copy.deepcopy(threshold_record),
            "0.5": copy.deepcopy(threshold_record),
            "0.6": copy.deepcopy(threshold_record),
        },
    }


def fixture_result() -> tuple[dict, dict, dict]:
    runs = {run: run_record(run) for run in validator.RUN_ORDER}
    selected = {run: record["thresholds"]["0.5"] for run, record in runs.items()}
    paired = {}
    for seed in (11, 23, 47):
        control = selected[f"control_seed{seed}"]
        gap = selected[f"gap8_seed{seed}"]
        fusion_delta = (
            gap["primary_pooled"]["conditional_fusion_rate"]
            - control["primary_pooled"]["conditional_fusion_rate"]
        )
        detection_delta = (
            gap["primary_pooled"]["site_center_detection_rate"]
            - control["primary_pooled"]["site_center_detection_rate"]
        )
        false_split_delta = (
            gap["single_sheet_control_pooled"]["false_split_rate"]
            - control["single_sheet_control_pooled"]["false_split_rate"]
        )
        paired[str(seed)] = {
            "conditional_fusion_delta": fusion_delta,
            "site_center_detection_delta": detection_delta,
            "false_split_delta": false_split_delta,
            "secondary_per_seed_diagnostics": {
                "fusion_reduction_at_least_10pp": fusion_delta <= -0.10,
                "detection_drop_no_more_than_2pp": detection_delta >= -0.02,
                "false_split_increase_no_more_than_2pp": false_split_delta <= 0.02,
            },
        }
    control_primary = validator.pool(
        [selected[f"control_seed{seed}"]["primary_pooled"] for seed in (11, 23, 47)],
        "fixture.pooled.control_primary",
    )
    gap_primary = validator.pool(
        [selected[f"gap8_seed{seed}"]["primary_pooled"] for seed in (11, 23, 47)],
        "fixture.pooled.gap_primary",
    )
    control_single = validator.pool(
        [selected[f"control_seed{seed}"]["single_sheet_control_pooled"] for seed in (11, 23, 47)],
        "fixture.pooled.control_single",
    )
    gap_single = validator.pool(
        [selected[f"gap8_seed{seed}"]["single_sheet_control_pooled"] for seed in (11, 23, 47)],
        "fixture.pooled.gap_single",
    )
    deltas = {
        "conditional_fusion_delta": gap_primary["conditional_fusion_rate"]
        - control_primary["conditional_fusion_rate"],
        "site_center_detection_delta": gap_primary["site_center_detection_rate"]
        - control_primary["site_center_detection_rate"],
        "false_split_delta": gap_single["false_split_rate"] - control_single["false_split_rate"],
    }
    pooled = {
        "control_primary": control_primary,
        "gap8_primary": gap_primary,
        "control_single_sheet_control": control_single,
        "gap8_single_sheet_control": gap_single,
        "deltas": deltas,
    }
    gates = {
        "pooled_conditional_fusion_reduction_at_least_10pp": (
            deltas["conditional_fusion_delta"] <= -0.10
        ),
        "negative_conditional_fusion_delta_all_three_seeds": all(
            record["conditional_fusion_delta"] < 0 for record in paired.values()
        ),
        "pooled_detection_drop_no_more_than_2pp": (deltas["site_center_detection_delta"] >= -0.02),
        "pooled_false_split_increase_no_more_than_2pp": (deltas["false_split_delta"] <= 0.02),
    }
    gates["synthetic_gate_pass"] = all(gates.values())
    result = {
        "schema_version": "1.1",
        "status": "sealed synthetic test scored once from complete ray caches",
        "gate_contract": "pre-inference clarification: pooled raw counts across matched seeds",
        "source_split_records_sha256": "split-records",
        "source_threshold_payload_sha256": "threshold-payload",
        "cell_count": 100,
        "runs": runs,
        "paired_gap8_minus_control": paired,
        "pooled_gap8_minus_control": pooled,
        "preregistered_synthetic_gates": gates,
        "payload_sha256": "fixture",
    }
    thresholds = {
        "payload_sha256": "threshold-payload",
        "runs": {run: {"selected_threshold": 0.5} for run in validator.RUN_ORDER},
    }
    split = {"records_sha256": "split-records"}
    return result, thresholds, split


def test_validator_accepts_pooled_primary_with_stricter_secondary_failure() -> None:
    result, thresholds, split = fixture_result()
    assert (
        result["paired_gap8_minus_control"]["47"]["secondary_per_seed_diagnostics"][
            "fusion_reduction_at_least_10pp"
        ]
        is False
    )
    assert result["preregistered_synthetic_gates"]["synthetic_gate_pass"] is True
    validator.validate_synthetic(result, thresholds, split)


def test_validator_rejects_altered_primary_gate() -> None:
    result, thresholds, split = fixture_result()
    result["preregistered_synthetic_gates"]["pooled_conditional_fusion_reduction_at_least_10pp"] = (
        False
    )
    with pytest.raises(RuntimeError, match="preregistered gates"):
        validator.validate_synthetic(result, thresholds, split)
