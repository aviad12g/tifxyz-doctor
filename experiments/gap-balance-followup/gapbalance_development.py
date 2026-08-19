"""Pure, result-blind rules for GapBalance development selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

SEEDS = (11, 23, 47)
ARMS = ("control", "gap2", "gap4")
RUNS = tuple(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS)
THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
SYNTHETIC_SEEDS = (400, 401, 402, 403)
PITCHES_UM = (170.0, 200.0, 230.0, 260.0)
PAPYRUS_LEVELS = (35, 50, 65, 90)
CONTROL_PITCH_UM = 700.0

COUNT_FIELDS = (
    "neighbour_sites",
    "detected_neighbour_sites",
    "fused_detected_sites",
    "control_sites",
    "false_split_sites",
)


def run_identity(run: str) -> tuple[str, int]:
    """Parse one of the nine frozen matched-run identities."""

    try:
        arm, seed_text = run.rsplit("_seed", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid run identity: {run}") from error
    if arm not in ARMS or seed not in SEEDS or run != f"{arm}_seed{seed}":
        raise ValueError(f"invalid run identity: {run}")
    return arm, seed


def choose_threshold(mean_blend: Mapping[float, float]) -> float:
    """Maximize mean blend; ties prefer nearest 0.5 and then lower."""

    if set(mean_blend) != set(THRESHOLDS):
        raise ValueError("mean-blend mapping must contain the exact frozen threshold grid")
    values = {float(key): float(value) for key, value in mean_blend.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("mean-blend values must be finite")
    return min(
        values,
        key=lambda threshold: (
            -values[threshold],
            abs(threshold - 0.5),
            threshold,
        ),
    )


def synthetic_cells() -> list[dict[str, Any]]:
    """Return the frozen 80-cell synthetic development grid."""

    cells: list[dict[str, Any]] = []
    for seed in SYNTHETIC_SEEDS:
        for pitch in PITCHES_UM:
            for papyrus in PAPYRUS_LEVELS:
                cells.append(
                    {
                        "kind": "primary",
                        "seed": seed,
                        "pitch_um": pitch,
                        "papyrus": papyrus,
                        "kollesis": True,
                        "name": f"primary_seed{seed}_pitch{int(pitch)}_pap{papyrus}",
                    }
                )
        for papyrus in PAPYRUS_LEVELS:
            cells.append(
                {
                    "kind": "single_sheet_control",
                    "seed": seed,
                    "pitch_um": CONTROL_PITCH_UM,
                    "papyrus": papyrus,
                    "kollesis": False,
                    "name": f"control_seed{seed}_pitch700_pap{papyrus}",
                }
            )
    if len(cells) != 80 or len({cell["name"] for cell in cells}) != 80:
        raise RuntimeError("frozen synthetic development grid changed")
    return cells


def _validated_counts(row: Mapping[str, Any]) -> dict[str, int]:
    if set(row) != set(COUNT_FIELDS):
        raise ValueError("synthetic count row has an unexpected schema")
    counts = {field: int(row[field]) for field in COUNT_FIELDS}
    if any(value < 0 or value != row[field] for field, value in counts.items()):
        raise ValueError("synthetic counts must be non-negative integers")
    if counts["detected_neighbour_sites"] > counts["neighbour_sites"]:
        raise ValueError("detected neighbour sites exceed neighbour sites")
    if counts["fused_detected_sites"] > counts["detected_neighbour_sites"]:
        raise ValueError("fused detected sites exceed detected neighbour sites")
    if counts["false_split_sites"] > counts["control_sites"]:
        raise ValueError("false split sites exceed control sites")
    return counts


def pool_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Pool raw denominators before computing preregistered rates."""

    if not rows:
        raise ValueError("cannot pool an empty count sequence")
    validated = [_validated_counts(row) for row in rows]
    totals = {
        field: sum(record[field] for record in validated)
        for field in COUNT_FIELDS
    }
    return {
        **totals,
        "site_center_detection_rate": (
            totals["detected_neighbour_sites"] / totals["neighbour_sites"]
            if totals["neighbour_sites"]
            else 0.0
        ),
        "conditional_fusion_rate": (
            totals["fused_detected_sites"] / totals["detected_neighbour_sites"]
            if totals["detected_neighbour_sites"]
            else 0.0
        ),
        "false_split_rate": (
            totals["false_split_sites"] / totals["control_sites"]
            if totals["control_sites"]
            else 0.0
        ),
    }


def _validate_selected_synthetic(
    selected_synthetic: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    if set(selected_synthetic) != set(RUNS):
        raise ValueError("synthetic result set must contain the exact nine runs")
    for run, record in selected_synthetic.items():
        run_identity(run)
        if set(record) != {"primary", "single_sheet_control"}:
            raise ValueError(f"{run}: synthetic result schema changed")
        _validated_counts(record["primary"])
        _validated_counts(record["single_sheet_control"])


def _validate_real_means(real_means: Mapping[str, Mapping[str, Any]]) -> None:
    if set(real_means) != set(RUNS):
        raise ValueError("real-development means must contain the exact nine runs")
    for run, record in real_means.items():
        run_identity(run)
        if set(record) != {"blend", "toposcore"}:
            raise ValueError(f"{run}: real-development mean schema changed")
        values = [float(record[key]) for key in ("blend", "toposcore")]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{run}: real-development means must be finite")


def summarize_arm(
    arm: str,
    selected_synthetic: Mapping[str, Mapping[str, Mapping[str, Any]]],
    real_means: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply all five pooled development eligibility constraints."""

    if arm not in {"gap2", "gap4"}:
        raise ValueError("candidate arm must be gap2 or gap4")
    _validate_selected_synthetic(selected_synthetic)
    _validate_real_means(real_means)

    control_primary = pool_counts(
        [selected_synthetic[f"control_seed{seed}"]["primary"] for seed in SEEDS]
    )
    candidate_primary = pool_counts(
        [selected_synthetic[f"{arm}_seed{seed}"]["primary"] for seed in SEEDS]
    )
    control_single = pool_counts(
        [
            selected_synthetic[f"control_seed{seed}"]["single_sheet_control"]
            for seed in SEEDS
        ]
    )
    candidate_single = pool_counts(
        [
            selected_synthetic[f"{arm}_seed{seed}"]["single_sheet_control"]
            for seed in SEEDS
        ]
    )

    conditional_fusion_reduction = float(
        control_primary["conditional_fusion_rate"]
        - candidate_primary["conditional_fusion_rate"]
    )
    detection_delta = float(
        candidate_primary["site_center_detection_rate"]
        - control_primary["site_center_detection_rate"]
    )
    false_split_delta = float(
        candidate_single["false_split_rate"] - control_single["false_split_rate"]
    )
    blend_deltas = [
        float(real_means[f"{arm}_seed{seed}"]["blend"])
        - float(real_means[f"control_seed{seed}"]["blend"])
        for seed in SEEDS
    ]
    toposcore_deltas = [
        float(real_means[f"{arm}_seed{seed}"]["toposcore"])
        - float(real_means[f"control_seed{seed}"]["toposcore"])
        for seed in SEEDS
    ]
    mean_blend_delta = sum(blend_deltas) / len(blend_deltas)
    mean_toposcore_delta = sum(toposcore_deltas) / len(toposcore_deltas)
    constraints = {
        "conditional_fusion_reduction_at_least_5pp": conditional_fusion_reduction >= 0.05,
        "detection_decline_no_worse_than_2pp": detection_delta >= -0.02,
        "false_split_increase_no_worse_than_2pp": false_split_delta <= 0.02,
        "mean_real_development_blend_delta_at_least_minus_0_005": mean_blend_delta
        >= -0.005,
        "mean_real_development_toposcore_delta_at_least_minus_0_005": mean_toposcore_delta
        >= -0.005,
    }
    return {
        "arm": arm,
        "pooled_counts": {
            "control_primary": control_primary,
            "candidate_primary": candidate_primary,
            "control_single_sheet": control_single,
            "candidate_single_sheet": candidate_single,
        },
        "deltas": {
            "conditional_fusion_reduction": conditional_fusion_reduction,
            "site_center_detection_delta": detection_delta,
            "false_split_delta": false_split_delta,
            "mean_real_development_blend_delta": mean_blend_delta,
            "mean_real_development_toposcore_delta": mean_toposcore_delta,
            "per_seed_real_blend_delta": dict(zip(map(str, SEEDS), blend_deltas, strict=True)),
            "per_seed_real_toposcore_delta": dict(
                zip(map(str, SEEDS), toposcore_deltas, strict=True)
            ),
        },
        "constraints": constraints,
        "eligible": all(constraints.values()),
    }


def select_candidate(
    selected_synthetic: Mapping[str, Mapping[str, Mapping[str, Any]]],
    real_means: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select exactly one eligible arm, or fail closed with no candidate."""

    summaries = {
        arm: summarize_arm(arm, selected_synthetic, real_means)
        for arm in ("gap2", "gap4")
    }
    eligible = [arm for arm, record in summaries.items() if record["eligible"]]
    if not eligible:
        selected = None
        reason = "neither arm satisfies every frozen development constraint"
    elif len(eligible) == 1:
        selected = eligible[0]
        reason = "only one arm satisfies every frozen development constraint"
    else:
        gap2_reduction = summaries["gap2"]["deltas"]["conditional_fusion_reduction"]
        gap4_reduction = summaries["gap4"]["deltas"]["conditional_fusion_reduction"]
        if abs(gap4_reduction - gap2_reduction) < 0.01:
            selected = "gap2"
            reason = "eligible reductions differ by less than 1pp; frozen tie-break selects Gap2"
        else:
            selected = "gap4" if gap4_reduction > gap2_reduction else "gap2"
            reason = "selected the eligible arm with the larger conditional-fusion reduction"
    return {
        "arm_summaries": summaries,
        "eligible_arms": eligible,
        "selected_arm": selected,
        "reason": reason,
        "confirmation_may_open": selected is not None,
    }
