#!/usr/bin/env python3
"""Freeze joint collection of sealed RunPod-real and Kaggle-synthetic results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect_runpod_real_kaggle_synthetic_results as collector
import orchestrate_heldout_queue as queue


STATUS = "RunPod real and Kaggle synthetic v4 mixed collection frozen before result access"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-predecessor-commit", required=True)
    parser.add_argument("--real-plan", type=Path, required=True)
    parser.add_argument("--real-receipt", type=Path, required=True)
    parser.add_argument("--real-output", type=Path, required=True)
    parser.add_argument("--synthetic-packages", type=Path, required=True)
    parser.add_argument("--synthetic-index", type=Path, required=True)
    parser.add_argument("--synthetic-receipt", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("mixed-collection plan output must start absent")
    if len(args.public_predecessor_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in args.public_predecessor_commit
    ):
        raise ValueError("public predecessor commit must be 40 lowercase hex")

    real_plan = queue.load_canonical(args.real_plan)
    real_receipt = queue.load_canonical(args.real_receipt)
    if (
        real_receipt.get("status")
        != "RunPod CPU real-scoring allocation terminated after sealed copy"
        or real_receipt.get("terminated_at") is None
        or real_receipt.get("scientific_outputs_inspected") is not False
        or real_receipt.get("plan_payload_sha256") != real_plan["payload_sha256"]
    ):
        raise RuntimeError("RunPod real receipt is not a sealed completed termination")
    ledger = collector.read_sha256s(args.real_output / "SHA256SUMS")
    collector.validate_real_source(
        args.real_output,
        expected_plan_payload=real_plan["payload_sha256"],
        expected_ledger=ledger,
    )
    synthetic = collector.validate_synthetic_source(
        args.synthetic_packages, args.synthetic_index, args.synthetic_receipt
    )
    expected_collector = Path(__file__).with_name(
        "collect_runpod_real_kaggle_synthetic_results.py"
    ).resolve()
    if args.collector.resolve() != expected_collector:
        raise RuntimeError("unexpected mixed collector path")

    payload = {
        "schema_version": "1.0",
        "status": STATUS,
        "public_predecessor_commit": args.public_predecessor_commit,
        "collector": collector.identity(args.collector),
        "runpod_real": {
            "execution_plan": {
                **collector.identity(args.real_plan),
                "payload_sha256": real_plan["payload_sha256"],
            },
            "provider_receipt": {
                **collector.identity(args.real_receipt),
                "payload_sha256": real_receipt["payload_sha256"],
            },
            "artifact_checksum_ledger": collector.identity(
                args.real_output / "SHA256SUMS"
            ),
            "artifact_sha256s": ledger,
            "source_kind": "sealed local copy from completed RunPod real scorer",
        },
        "kaggle_synthetic_v4": {
            "kernel_id": collector.SYNTHETIC_KERNEL,
            "kernel_version": 4,
            "package_index": synthetic["index_identity"],
            "acceptance_receipt": synthetic["receipt_identity"],
            "launcher": synthetic["launcher_identity"],
            "source_kind": "completed sealed Kaggle synthetic scorer version 4",
        },
        "collection_contract": {
            "both_sources_verified_before_output_creation": True,
            "real_source_is_copied_without_parsing": True,
            "synthetic_download_is_selected_and_log_suppressed": True,
            "result_json_scoring_manifest_or_panel_parsed_before_joint_collection": False,
            "scientific_outputs_opened_or_inspected": False,
            "independent_validation_required_after_collection": True,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("mixed collection plan payload SHA-256:", payload["payload_sha256"])
    print("RUNPOD_KAGGLE_MIXED_COLLECTION_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
