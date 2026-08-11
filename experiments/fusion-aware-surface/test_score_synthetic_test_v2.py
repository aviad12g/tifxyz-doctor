from score_synthetic_test_v2 import pool_counts, summarize_selected


def record(*, neighbours: int, detected: int, fused: int, controls: int, splits: int) -> dict:
    return pool_counts(
        [
            {
                "neighbour_sites": neighbours,
                "detected_neighbour_sites": detected,
                "fused_detected_sites": fused,
                "control_sites": controls,
                "false_split_sites": splits,
            }
        ]
    )


def selected(primary: dict, control: dict) -> dict:
    return {
        "primary_pooled": primary,
        "single_sheet_control_pooled": control,
    }


def test_primary_gates_pool_raw_counts_but_require_each_fusion_sign() -> None:
    values = {}
    controls = {
        11: record(neighbours=100, detected=100, fused=50, controls=100, splits=5),
        23: record(neighbours=100, detected=100, fused=50, controls=100, splits=5),
        47: record(neighbours=100, detected=100, fused=50, controls=100, splits=5),
    }
    gaps = {
        11: record(neighbours=100, detected=99, fused=35, controls=100, splits=6),
        23: record(neighbours=100, detected=99, fused=39, controls=100, splits=6),
        47: record(neighbours=100, detected=99, fused=42, controls=100, splits=6),
    }
    for seed in (11, 23, 47):
        values[f"control_seed{seed}"] = selected(controls[seed], controls[seed])
        values[f"gap8_seed{seed}"] = selected(gaps[seed], gaps[seed])

    paired, pooled, gates = summarize_selected(values)

    assert pooled["control_primary"]["conditional_fusion_rate"] == 0.5
    assert pooled["gap8_primary"]["conditional_fusion_rate"] == 116 / 297
    assert gates == {
        "pooled_conditional_fusion_reduction_at_least_10pp": True,
        "negative_conditional_fusion_delta_all_three_seeds": True,
        "pooled_detection_drop_no_more_than_2pp": True,
        "pooled_false_split_increase_no_more_than_2pp": True,
        "synthetic_gate_pass": True,
    }
    assert paired["47"]["secondary_per_seed_diagnostics"]["fusion_reduction_at_least_10pp"] is False


def test_primary_fusion_gate_requires_negative_delta_in_every_seed() -> None:
    values = {}
    for seed, gap_fused in ((11, 20), (23, 20), (47, 50)):
        base = record(neighbours=100, detected=100, fused=50, controls=100, splits=5)
        gap = record(
            neighbours=100,
            detected=100,
            fused=gap_fused,
            controls=100,
            splits=5,
        )
        values[f"control_seed{seed}"] = selected(base, base)
        values[f"gap8_seed{seed}"] = selected(gap, gap)

    _, _, gates = summarize_selected(values)
    assert gates["pooled_conditional_fusion_reduction_at_least_10pp"] is True
    assert gates["negative_conditional_fusion_delta_all_three_seeds"] is False
    assert gates["synthetic_gate_pass"] is False
