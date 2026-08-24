#!/usr/bin/env python3
"""Score sealed synthetic ray caches using the clarified preregistered gates.

The frozen prose defines pooled magnitude gates across the three matched
training seeds, plus a separate requirement that the fusion delta be negative
in every seed.  The original ``score_synthetic_test.py`` applied all three
numeric tolerances independently to every seed.  This pre-inference scorer
keeps those stricter per-seed checks as secondary diagnostics while using raw
count pooling for the primary gates described in ``PREREGISTRATION.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cache_real_predictions import RUNS, sha256_file
from cache_synthetic_rays import synthetic_cells, validate_ray_cache
from fusion_ray_readout import score_rays
from score_real_test import SENSITIVITY_THRESHOLDS, validate_embedded_hash

COUNT_FIELDS = (
    "neighbour_sites",
    "detected_neighbour_sites",
    "fused_detected_sites",
    "control_sites",
    "false_split_sites",
)
SEEDS = (11, 23, 47)


def pool_counts(rows: list[dict]) -> dict[str, int | float]:
    totals = {field: int(sum(int(row[field]) for row in rows)) for field in COUNT_FIELDS}
    return {
        **totals,
        "site_center_detection_rate": totals["detected_neighbour_sites"]
        / max(totals["neighbour_sites"], 1),
        "conditional_fusion_rate": totals["fused_detected_sites"]
        / max(totals["detected_neighbour_sites"], 1),
        "false_split_rate": totals["false_split_sites"] / max(totals["control_sites"], 1),
    }


def validate_run(
    root: Path,
    run: str,
    expected: dict[str, dict],
    split_sha256: str,
    threshold: float,
) -> dict[str, Path]:
    observed = {}
    for shard in range(10):
        path = root / f"synthetic_manifest_{run}_shard{shard:02d}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_embedded_hash(manifest)
        if manifest.get("status") != "sealed synthetic rays cached; endpoints not scored":
            raise RuntimeError(f"{run} shard {shard}: wrong status")
        if manifest.get("run") != run or manifest.get("shard_index") != shard:
            raise RuntimeError(f"{run} shard {shard}: wrong identity")
        if manifest.get("shard_count") != 10:
            raise RuntimeError(f"{run} shard {shard}: wrong shard count")
        if manifest.get("source_split_records_sha256") != split_sha256:
            raise RuntimeError(f"{run} shard {shard}: wrong split")
        if float(manifest.get("selected_threshold")) != threshold:
            raise RuntimeError(f"{run} shard {shard}: wrong frozen threshold")
        for record in manifest.get("cells", []):
            name = record.get("name")
            if name not in expected or name in observed:
                raise RuntimeError(f"{run}: duplicate or unexpected cell {name}")
            if any(record.get(key) != expected[name].get(key) for key in expected[name]):
                raise RuntimeError(f"{run}: changed cell design for {name}")
            cache = root / run / record["file"]
            validate_ray_cache(cache)
            if sha256_file(cache) != record["sha256"] or cache.stat().st_size != record["bytes"]:
                raise RuntimeError(f"{run}: ray-cache hash/size mismatch for {name}")
            observed[name] = cache
    if set(observed) != set(expected):
        raise RuntimeError(f"{run}: incomplete synthetic test grid")
    return observed


def summarize_selected(selected_pooled: dict[str, dict]) -> tuple[dict, dict, dict]:
    """Recompute paired diagnostics and the prose-defined pooled primary gates."""

    paired = {}
    for seed in SEEDS:
        control = selected_pooled[f"control_seed{seed}"]
        gap = selected_pooled[f"gap8_seed{seed}"]
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

    control_primary = pool_counts(
        [selected_pooled[f"control_seed{seed}"]["primary_pooled"] for seed in SEEDS]
    )
    gap_primary = pool_counts(
        [selected_pooled[f"gap8_seed{seed}"]["primary_pooled"] for seed in SEEDS]
    )
    control_single = pool_counts(
        [selected_pooled[f"control_seed{seed}"]["single_sheet_control_pooled"] for seed in SEEDS]
    )
    gap_single = pool_counts(
        [selected_pooled[f"gap8_seed{seed}"]["single_sheet_control_pooled"] for seed in SEEDS]
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
            record["conditional_fusion_delta"] < 0.0 for record in paired.values()
        ),
        "pooled_detection_drop_no_more_than_2pp": (deltas["site_center_detection_delta"] >= -0.02),
        "pooled_false_split_increase_no_more_than_2pp": (deltas["false_split_delta"] <= 0.02),
    }
    gates["synthetic_gate_pass"] = all(gates.values())
    return paired, pooled, gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--threshold-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    thresholds = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
    validate_embedded_hash(thresholds)
    if (
        thresholds.get("status")
        != "thresholds frozen from Scroll-1 validation before test inference"
    ):
        raise RuntimeError("threshold manifest has the wrong status")
    if thresholds.get("source_split_records_sha256") != split.get("records_sha256"):
        raise RuntimeError("threshold and split manifests disagree")
    expected = {cell["name"]: cell for cell in synthetic_cells()}

    run_results = {}
    selected_pooled = {}
    for run in RUNS:
        selected = float(thresholds["runs"][run]["selected_threshold"])
        caches = validate_run(args.ray_root, run, expected, split["records_sha256"], selected)
        tested = sorted(set(SENSITIVITY_THRESHOLDS + (selected,)))
        threshold_results = {}
        for threshold in tested:
            rows = {}
            for name, path in sorted(caches.items()):
                with np.load(path) as data:
                    rows[name] = score_rays(data["ray_probability"], data["ray_turn"], threshold)
            primary = pool_counts(
                [rows[name] for name, cell in expected.items() if cell["kind"] == "primary"]
            )
            controls = pool_counts(
                [
                    rows[name]
                    for name, cell in expected.items()
                    if cell["kind"] == "single_sheet_control"
                ]
            )
            threshold_results[str(threshold)] = {
                "primary_pooled": primary,
                "single_sheet_control_pooled": controls,
                "per_cell": rows,
            }
            print(
                f"SYNTHETIC_SCORE run={run} threshold={threshold:.2f} "
                f"detection={primary['site_center_detection_rate']:.6f} "
                f"conditional_fusion={primary['conditional_fusion_rate']:.6f} "
                f"false_split={controls['false_split_rate']:.6f}",
                flush=True,
            )
        selected_pooled[run] = threshold_results[str(selected)]
        run_results[run] = {
            "selected_threshold": selected,
            "thresholds": threshold_results,
        }

    paired, pooled, gates = summarize_selected(selected_pooled)
    payload = {
        "schema_version": "1.1",
        "status": "sealed synthetic test scored once from complete ray caches",
        "gate_contract": "pre-inference clarification: pooled raw counts across matched seeds",
        "source_split_records_sha256": split["records_sha256"],
        "source_threshold_payload_sha256": thresholds["payload_sha256"],
        "cell_count": len(expected),
        "runs": run_results,
        "paired_gap8_minus_control": paired,
        "pooled_gap8_minus_control": pooled,
        "preregistered_synthetic_gates": gates,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("synthetic gates:", gates)
    print("payload SHA-256:", payload["payload_sha256"])
    print("SEALED_SYNTHETIC_TEST_SCORED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
