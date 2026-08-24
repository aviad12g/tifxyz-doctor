#!/usr/bin/env python3
"""Launch only the result-blind accelerated real retry and retain synthetic v4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import orchestrate_heldout_queue as queue
import orchestrate_scoring_pair as pair


PUBLIC_PLAN_COMMIT = "9940f7db306e15398dcbe9ed44e270ac0d6d41e1"
PUBLIC_DELIVERY_COMMIT = "be34445c6ca3b1a82f213e0cce05a2379b30c14c"
TARGET_REAL_VERSION = 6
REAL_KERNEL = "aviadcohen1/vesuvius-fusion-one-shot-real-scoring"
SYNTHETIC_KERNEL = "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring"


def sha256_file(path: Path) -> str:
    return queue.sha256_file(path)


def load_v4_receipt(path: Path) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    accepted = receipt.get("accepted")
    expected = [
        {"mode": "real", "kernel_id": REAL_KERNEL, "kernel_version": 4},
        {"mode": "synthetic", "kernel_id": SYNTHETIC_KERNEL, "kernel_version": 4},
    ]
    if not isinstance(accepted, list) or [
        {key: record.get(key) for key in ("mode", "kernel_id", "kernel_version")}
        for record in accepted
    ] != expected:
        raise RuntimeError("v4 receipt does not bind the recorded paired attempt")
    return receipt


def validate_inputs(index: dict, packages: list[dict], v4_receipt: dict) -> dict:
    if index.get("public_execution_plan", {}).get("commit") != PUBLIC_PLAN_COMMIT:
        raise RuntimeError("retry package index points to another public plan")
    if index.get("public_cache_delivery", {}).get("commit") != PUBLIC_DELIVERY_COMMIT:
        raise RuntimeError("retry package index points to another public delivery")
    by_mode = {record["mode"]: record for record in packages}
    if set(by_mode) != {"real", "synthetic"}:
        raise RuntimeError("retry package mode set mismatch")
    if by_mode["real"]["kaggle_kernel_id"] != REAL_KERNEL:
        raise RuntimeError("real retry kernel identity mismatch")
    if by_mode["synthetic"]["kaggle_kernel_id"] != SYNTHETIC_KERNEL:
        raise RuntimeError("synthetic retained kernel identity mismatch")
    return by_mode["real"]


def load_failed_v5_receipt(path: Path) -> dict:
    receipt = queue.load_canonical(path)
    real = receipt.get("real_retry", {})
    if (
        receipt.get("status")
        != "accelerated real v5 accepted; completed synthetic v4 retained sealed"
        or real.get("kernel_id") != REAL_KERNEL
        or real.get("kernel_version") != 5
        or receipt.get("retained_synthetic", {}).get("kernel_version") != 4
    ):
        raise RuntimeError("failed-v5 receipt identity mismatch")
    return receipt


def receipt_payload(
    *,
    index_path: Path,
    index: dict,
    v4_receipt_path: Path,
    failed_v5_receipt_path: Path,
    real_package: dict,
    version: int,
) -> dict:
    controller = Path(__file__).resolve()
    return {
        "schema_version": "1.0",
        "status": "accelerated real v6 accepted; completed synthetic v4 retained sealed",
        "package_index": {
            "file": index_path.name,
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "payload_sha256": index["payload_sha256"],
        },
        "public_execution_plan_commit": PUBLIC_PLAN_COMMIT,
        "public_cache_delivery_commit": PUBLIC_DELIVERY_COMMIT,
        "predecessor_v4_receipt": {
            "file": v4_receipt_path.name,
            "bytes": v4_receipt_path.stat().st_size,
            "sha256": sha256_file(v4_receipt_path),
        },
        "failed_v5_receipt": {
            "file": failed_v5_receipt_path.name,
            "bytes": failed_v5_receipt_path.stat().st_size,
            "sha256": sha256_file(failed_v5_receipt_path),
        },
        "real_retry": {
            "mode": "real",
            "kernel_id": REAL_KERNEL,
            "kernel_version": version,
            "package_ledger_sha256": real_package["files"]["KERNEL_SHA256SUMS"]["sha256"],
            "accepted_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
        "retained_synthetic": {
            "mode": "synthetic",
            "kernel_id": SYNTHETIC_KERNEL,
            "kernel_version": 4,
            "status": "COMPLETE",
            "result_opened_downloaded_or_used": False,
        },
        "controller": {
            "file": controller.name,
            "bytes": controller.stat().st_size,
            "sha256": sha256_file(controller),
        },
        "scientific_gate": {
            "real_v4_result_opened_downloaded_or_used": False,
            "real_v5_result_created_opened_downloaded_or_used": False,
            "synthetic_v4_result_opened_downloaded_or_used": False,
            "synthetic_scientific_scorer_repeated": False,
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "real_retry_runs_panels_before_parallel_scoring": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v4-receipt", type=Path, required=True)
    parser.add_argument("--failed-v5-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError("accelerated retry receipt must start absent")
    index, packages = pair.validate_packages(args.packages, args.index)
    v4_receipt = load_v4_receipt(args.v4_receipt)
    load_failed_v5_receipt(args.failed_v5_receipt)
    real_package = validate_inputs(index, packages, v4_receipt)
    if queue.kernel_status(args.kaggle, REAL_KERNEL) != "ERROR":
        raise RuntimeError("real v4 is not the recorded ERROR predecessor")
    if queue.kernel_status(args.kaggle, SYNTHETIC_KERNEL) != "COMPLETE":
        raise RuntimeError("synthetic v4 is not COMPLETE and sealed")
    version = pair.push_package(args.kaggle, args.packages, real_package)
    if version != TARGET_REAL_VERSION:
        raise RuntimeError("accelerated real retry did not create version 6")
    payload = receipt_payload(
        index_path=args.index,
        index=index,
        v4_receipt_path=args.v4_receipt,
        failed_v5_receipt_path=args.failed_v5_receipt,
        real_package=real_package,
        version=version,
    )
    pair.write_receipt(args.receipt, payload)
    print(
        "ACCELERATED_REAL_RETRY_ACCEPTED version=6 "
        f"url=https://www.kaggle.com/code/{REAL_KERNEL}",
        flush=True,
    )
    print("SYNTHETIC_V4_RETAINED_SEALED_WITHOUT_RERUN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
