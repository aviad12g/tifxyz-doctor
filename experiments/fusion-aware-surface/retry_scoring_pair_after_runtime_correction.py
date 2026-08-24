#!/usr/bin/env python3
"""Launch scorer v4 after the recorded result-blind runtime failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import collect_heldout_delivery_inputs as heldout
import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair


PUBLIC_PLAN_COMMIT = "624c1ef41fa078ea9b3d3d97207a56ab6662e27b"
PUBLIC_DELIVERY_COMMIT = "367050ad1ff23f97be7996fffb978220678bd91c"
EXPECTED_PROVIDER_VERSION = 4
KERNEL_IDS = {
    "real": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
    "synthetic": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
}


def expected_attempts(version: int) -> list[dict]:
    return [
        {"mode": mode, "kernel_id": KERNEL_IDS[mode], "kernel_version": version}
        for mode in pair.MODES
    ]


def validate_correction_plan(path: Path, index: dict) -> dict:
    plan = queue.load_canonical(path)
    if index.get("public_execution_plan") != {
        "commit": PUBLIC_PLAN_COMMIT,
        "file_sha256": queue.sha256_file(path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("retry packages do not bind the final public runtime plan")
    if index.get("public_cache_delivery", {}).get("commit") != PUBLIC_DELIVERY_COMMIT:
        raise RuntimeError("retry packages do not bind the final public runtime delivery")

    asset = plan.get("result_blind_scoring_asset_correction", {})
    compatibility = plan.get(
        "result_blind_scoring_job_plan_compatibility_correction", {}
    )
    runtime = plan.get("result_blind_scoring_runtime_transport_correction", {})
    projection = plan.get("result_blind_scoring_runtime_projection_correction", {})
    if asset.get("failed_attempts") != expected_attempts(1):
        raise RuntimeError("public plan does not record the v1 failures")
    if compatibility.get("failed_attempts") != expected_attempts(2):
        raise RuntimeError("public plan does not record the v2 failures")
    if runtime.get("failed_attempts") != expected_attempts(3):
        raise RuntimeError("public plan does not record the v3 failures")
    if runtime.get("correction") != {
        "torch": "2.5.1+cpu",
        "torchvision": "0.20.1+cpu",
        "other_runtime_versions_changed": False,
        "primary_index": "https://download.pytorch.org/whl/cpu",
        "extra_index": "https://pypi.org/simple",
        "binary_wheels_only": True,
        "scorer_kernel_accelerator": "none",
        "cache_manifest_or_npz_changed": False,
    }:
        raise RuntimeError("public CPU runtime correction differs from the freeze")
    resolution = runtime.get("python312_linux_resolution_preflight", {})
    if (
        resolution.get("status")
        != "all exact top-level requirements and transitive wheels resolved"
        or resolution.get("files") != 30
        or resolution.get("bytes") != 260266954
        or resolution.get("sorted_wheel_ledger_sha256")
        != "7e8967a004c7e6e97a1e223f33f8edb78731c1d1dfa7c21ea52df9c63747bfec"
    ):
        raise RuntimeError("public Python 3.12 wheel preflight differs from the freeze")
    if (
        runtime.get("scientific_gate", {}).get("scientific_result_created_or_opened")
        is not False
        or runtime.get("scientific_gate", {}).get(
            "held_out_input_staging_started_in_failed_attempts"
        )
        is not False
        or runtime.get("scientific_gate", {}).get(
            "scorer_invocation_started_in_failed_attempts"
        )
        is not False
        or projection.get("local_preflight", {}).get("provider_scorer_version_created")
        is not False
        or projection.get("scientific_gate", {}).get(
            "scientific_result_created_or_opened"
        )
        is not False
    ):
        raise RuntimeError("public result-blind runtime gates differ from the freeze")
    return plan


def validate_predecessors(kaggle: str, packages: list[dict]) -> None:
    for package in packages:
        mode = package["mode"]
        if package["kaggle_kernel_id"] != KERNEL_IDS[mode]:
            raise RuntimeError(f"{mode}: retry kernel identity mismatch")
        status = queue.kernel_status(kaggle, KERNEL_IDS[mode])
        version, _sealed_source = heldout.current_kernel_version_and_source(
            KERNEL_IDS[mode]
        )
        if status != "ERROR" or version != 3:
            raise RuntimeError(f"{mode}: predecessor is not the recorded failed v3")


def new_receipt(index_path: Path, index: dict) -> dict:
    receipt = pair.empty_receipt(index_path, index)
    receipt["predecessor_fail_closed_attempts"] = [
        {
            **record,
            "status": "ERROR",
            "scientific_result_created_or_opened": False,
        }
        for version in (1, 2, 3)
        for record in expected_attempts(version)
    ]
    controller = Path(__file__).resolve()
    receipt["result_blind_retry_controller"] = {
        "file": controller.name,
        "bytes": controller.stat().st_size,
        "sha256": queue.sha256_file(controller),
    }
    receipt["scientific_gate"] = {
        "v1_v2_v3_failed_before_scorer_invocation": True,
        "v1_v2_v3_scientific_results_created_or_opened": False,
        "full_local_synthetic_staging_preflight_passed": True,
        "exact_python312_cpu_wheel_resolution_passed": True,
        "v4_pair_must_be_accepted_before_result_access": True,
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
            f"SCORING_V4_ACCEPTED mode={package['mode']} version={version} "
            f"url=https://www.kaggle.com/code/{package['kaggle_kernel_id']}",
            flush=True,
        )
    if [record["mode"] for record in receipt["accepted"]] != list(pair.MODES):
        raise RuntimeError("both v4 scorer versions were not accepted")
    print("BOTH_V4_ONE_SHOT_SCORERS_ACCEPTED_BEFORE_RESULT_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
