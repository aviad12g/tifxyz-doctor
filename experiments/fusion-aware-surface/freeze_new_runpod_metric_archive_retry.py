#!/usr/bin/env python3
"""Freeze the exact two-directory public metric archive correction."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "e53428e14e429d353a28acc8b0f7618cbef20778a04b02751b3a36da7478e59b"
FAILED_RECEIPT_PAYLOAD = "8ecd5153f7b95c396f6950ab7e91d4e2c760741a199f3fd6e799be9bee49f6fe"


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
        raise RuntimeError("metric-archive retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong rsync-corrected predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle verified-parallel pipeline started"
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind metric-archive failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU parallel metric-archive-layout retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_wrapper"] = identity(args.wrapper)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_fresh_retry_operational_allowance_usd": 0.25,
            "compute_cutoff_usd": 19.75,
            "maximum_provider_wall_seconds_at_price_ceiling": 55_546,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 55_246,
        }
    )
    plan["result_blind_public_metric_archive_layout_retry"] = {
        "schema_version": "1.0",
        "status": "exact two-directory public metric archive correction frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "failure_state": "PREPARING_PUBLIC_RUNTIME",
            "failure_message": "public metric archive layout mismatch",
            "credentials_removed": True,
            "private_dataset_pull_started": False,
            "real_scorer_started": False,
        },
        "correction": {
            "required_top_level_directories": [
                "topological-metrics-kaggle",
                "wheels",
            ],
            "unused_bundled_wheels_removed": True,
            "exact_metric_source_retained": True,
            "separately_frozen_runtime_wheels_retained": True,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "metric_source_bytes_changed": False,
            "metric_runtime_versions_changed": False,
            "cache_model_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "held_out_result_opened_or_used": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_METRIC_ARCHIVE_LAYOUT_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
