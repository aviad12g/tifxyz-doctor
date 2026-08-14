#!/usr/bin/env python3
"""Freeze the exact KaggleHub completion-marker transport correction."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "236d593e5357cc45fbf0b897b981ac0c8b4695d434435237797dd0848ea01d9d"
FAILED_RECEIPT_PAYLOAD = "5d0325a2e921e5f8a749c0d5e6b36eee418a1ba412c87c43fc1e61d13c07e1e9"
EXPECTED_MARKER = (
    ".complete/datasets/aviadcohen1/"
    "vesuvius-fusion-real-heldout-transport-v1/1/bundle.complete"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("payload_sha256") != canonical_sha256(payload):
        raise RuntimeError(f"invalid hashed JSON: {path}")
    return payload


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("public tooling commit must be 40 lowercase hexadecimal characters")
    return value


def write_hashed(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError("completion-marker retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--puller", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong metric-archive-corrected predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle verified-parallel pipeline started"
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind private-transport failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU KaggleHub completion-marker retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_puller"] = identity(args.puller)
    plan["private_transport_wrapper"] = identity(args.wrapper)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_metric_archive_retry_operational_allowance_usd": 0.75,
            "compute_cutoff_usd": 19.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 53_437,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 53_137,
        }
    )
    plan["result_blind_kagglehub_completion_marker_retry"] = {
        "schema_version": "1.0",
        "status": "exact pinned KaggleHub completion-marker correction frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "pipeline_entered_private_transport": True,
            "real_scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "correction": {
            "pinned_kagglehub_version": "1.0.2",
            "exact_expected_non_dataset_marker": EXPECTED_MARKER,
            "expected_marker_bytes": 0,
            "reject_any_other_completion_entry": True,
            "remove_only_completion_tree_before_dataset_inventory": True,
            "propagate_result_blind_child_error": True,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "private_dataset_files_changed": False,
            "private_dataset_expected_files": 284,
            "private_source_files": 281,
            "npz_payloads_opened_or_parsed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_KAGGLEHUB_COMPLETION_MARKER_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
