#!/usr/bin/env python3
"""Relaunch the frozen scorer pair after the recorded pre-staging v1 failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import collect_heldout_delivery_inputs as heldout
import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair


PUBLIC_PLAN_COMMIT = "393dee840bf4ef6c75f228023d089a8ea22c8785"
PUBLIC_DELIVERY_COMMIT = "3c8722ccf6d13b046367496af785a99a7ce1c5af"
FAILED_ATTEMPTS = {
    "real": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
        "kernel_version": 1,
    },
    "synthetic": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
        "kernel_version": 1,
    },
}


def validate_correction_plan(path: Path, index: dict) -> dict:
    plan = queue.load_canonical(path)
    binding = index.get("public_execution_plan")
    if binding != {
        "commit": PUBLIC_PLAN_COMMIT,
        "file_sha256": queue.sha256_file(path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("retry packages do not bind the public correction plan")
    if index.get("public_cache_delivery", {}).get("commit") != PUBLIC_DELIVERY_COMMIT:
        raise RuntimeError("retry packages do not bind the public corrected delivery")
    correction = plan.get("result_blind_scoring_asset_correction", {})
    expected_failures = [
        {"mode": mode, **FAILED_ATTEMPTS[mode]} for mode in pair.MODES
    ]
    if (
        correction.get("failed_attempts") != expected_failures
        or correction.get("failure_stage")
        != "asset-ledger validation before held-out input staging"
        or correction.get("scientific_gate")
        != {
            "held_out_input_staging_started_in_failed_attempts": False,
            "scorer_invocation_started_in_failed_attempts": False,
            "cache_npz_opened_or_inspected": False,
            "scientific_result_created_or_opened": False,
            "threshold_seed_panel_gate_or_claim_changed": False,
            "retry_is_first_scientific_scorer_invocation": True,
        }
    ):
        raise RuntimeError("public fail-closed retry authorization differs from the freeze")
    return plan


def new_receipt(index_path: Path, index: dict) -> dict:
    receipt = pair.empty_receipt(index_path, index)
    receipt["predecessor_fail_closed_attempts"] = [
        {"mode": mode, **FAILED_ATTEMPTS[mode], "status": "ERROR"}
        for mode in pair.MODES
    ]
    controller = Path(__file__).resolve()
    receipt["result_blind_retry_controller"] = {
        "file": controller.name,
        "bytes": controller.stat().st_size,
        "sha256": queue.sha256_file(controller),
    }
    receipt["scientific_gate"] = {
        "v1_failed_before_input_staging": True,
        "v1_scientific_results_created_or_opened": False,
        "v2_pair_must_be_accepted_before_result_access": True,
        "scientific_contract_changed": False,
    }
    return receipt


def validate_predecessors(kaggle: str, packages: list[dict]) -> None:
    for package in packages:
        mode = package["mode"]
        expected = FAILED_ATTEMPTS[mode]
        if package["kaggle_kernel_id"] != expected["kernel_id"]:
            raise RuntimeError(f"{mode}: retry kernel identity mismatch")
        status = queue.kernel_status(kaggle, expected["kernel_id"])
        version, _sealed_source = heldout.current_kernel_version_and_source(
            expected["kernel_id"]
        )
        if status != "ERROR" or version != expected["kernel_version"]:
            raise RuntimeError(f"{mode}: predecessor is not the recorded failed v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    index, packages = pair.validate_packages(args.packages, args.index)
    validate_correction_plan(args.plan, index)
    if args.receipt.exists():
        raise RuntimeError("result-blind retry receipt must start absent")
    validate_predecessors(args.kaggle, packages)
    receipt = new_receipt(args.index, index)
    for package in packages:
        version = pair.push_package(args.kaggle, args.packages, package)
        if version != 2:
            raise RuntimeError(f"{package['mode']}: retry did not create version 2")
        receipt["accepted"].append(
            {
                "mode": package["mode"],
                "kernel_id": package["kaggle_kernel_id"],
                "kernel_version": version,
                "package_ledger_sha256": package["files"]["KERNEL_SHA256SUMS"][
                    "sha256"
                ],
                "accepted_at_utc": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
            }
        )
        pair.write_receipt(args.receipt, receipt)
        print(
            f"SCORING_RETRY_ACCEPTED mode={package['mode']} version={version} "
            f"url=https://www.kaggle.com/code/{package['kaggle_kernel_id']}",
            flush=True,
        )
    if [record["mode"] for record in receipt["accepted"]] != list(pair.MODES):
        raise RuntimeError("both corrected scorer versions were not accepted")
    print("BOTH_CORRECTED_ONE_SHOT_SCORERS_ACCEPTED_BEFORE_RESULT_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
