#!/usr/bin/env python3
"""Launch scorer v3 after the two recorded result-blind staging failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import collect_heldout_delivery_inputs as heldout
import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair


PUBLIC_PLAN_COMMIT = "e36169e53acbf83ae77642dc47b00c6f98a24ec1"
PUBLIC_DELIVERY_COMMIT = "7fe1c06671b56c66554ca377d4808af3cf76187c"
EXPECTED_PROVIDER_VERSION = 3
FAILED_ATTEMPTS = {
    "real": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
        "versions": [1, 2],
    },
    "synthetic": {
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
        "versions": [1, 2],
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
        raise RuntimeError("retry packages do not bind the final public correction plan")
    if index.get("public_cache_delivery", {}).get("commit") != PUBLIC_DELIVERY_COMMIT:
        raise RuntimeError("retry packages do not bind the final public corrected delivery")

    asset = plan.get("result_blind_scoring_asset_correction", {})
    compatibility = plan.get(
        "result_blind_scoring_job_plan_compatibility_correction", {}
    )
    schema = plan.get("result_blind_scoring_cache_identity_schema_correction", {})
    expected_v1 = [
        {
            "mode": mode,
            "kernel_id": FAILED_ATTEMPTS[mode]["kernel_id"],
            "kernel_version": 1,
        }
        for mode in pair.MODES
    ]
    expected_v2 = [
        {
            "mode": mode,
            "kernel_id": FAILED_ATTEMPTS[mode]["kernel_id"],
            "kernel_version": 2,
        }
        for mode in pair.MODES
    ]
    if asset.get("failed_attempts") != expected_v1:
        raise RuntimeError("public correction plan does not record the v1 failures")
    if compatibility.get("failed_attempts") != expected_v2:
        raise RuntimeError("public correction plan does not record the v2 failures")
    if (
        asset.get("scientific_gate", {}).get("scientific_result_created_or_opened")
        is not False
        or compatibility.get("scientific_gate", {}).get(
            "scientific_result_created_or_opened"
        )
        is not False
        or compatibility.get("scientific_gate", {}).get("scorer_invocation_started")
        is not False
        or schema.get("local_preflight", {}).get("provider_scorer_version_created")
        is not False
        or schema.get("scientific_gate", {}).get(
            "scientific_result_created_or_opened"
        )
        is not False
        or schema.get("scientific_gate", {}).get(
            "full_synthetic_staging_preflight_required_before_provider_retry"
        )
        is not True
    ):
        raise RuntimeError("public result-blind scientific gates differ from the freeze")
    return plan


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
        if status != "ERROR" or version != expected["versions"][-1]:
            raise RuntimeError(f"{mode}: predecessor is not the recorded failed v2")


def new_receipt(index_path: Path, index: dict) -> dict:
    receipt = pair.empty_receipt(index_path, index)
    receipt["predecessor_fail_closed_attempts"] = [
        {
            "mode": mode,
            "kernel_id": FAILED_ATTEMPTS[mode]["kernel_id"],
            "kernel_version": version,
            "status": "ERROR",
            "scientific_result_created_or_opened": False,
        }
        for version in (1, 2)
        for mode in pair.MODES
    ]
    controller = Path(__file__).resolve()
    receipt["result_blind_retry_controller"] = {
        "file": controller.name,
        "bytes": controller.stat().st_size,
        "sha256": queue.sha256_file(controller),
    }
    receipt["scientific_gate"] = {
        "v1_and_v2_failed_before_scorer_invocation": True,
        "v1_and_v2_scientific_results_created_or_opened": False,
        "full_local_synthetic_staging_preflight_passed": True,
        "v3_pair_must_be_accepted_before_result_access": True,
        "scientific_contract_changed": False,
    }
    return receipt


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
        if version != EXPECTED_PROVIDER_VERSION:
            raise RuntimeError(
                f"{package['mode']}: retry did not create provider version "
                f"{EXPECTED_PROVIDER_VERSION}"
            )
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
            f"SCORING_V3_ACCEPTED mode={package['mode']} version={version} "
            f"url=https://www.kaggle.com/code/{package['kaggle_kernel_id']}",
            flush=True,
        )
    if [record["mode"] for record in receipt["accepted"]] != list(pair.MODES):
        raise RuntimeError("both v3 scorer versions were not accepted")
    print("BOTH_V3_ONE_SHOT_SCORERS_ACCEPTED_BEFORE_RESULT_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
