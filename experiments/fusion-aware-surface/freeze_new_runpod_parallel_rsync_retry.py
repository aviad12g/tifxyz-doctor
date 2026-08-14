#!/usr/bin/env python3
"""Freeze the public-rsync correction for a fresh parallel CPU retry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "a9ed924742a2f056e9493d986df8ec0a0220fe5ff358d8863d5f5c5144652529"
FAILED_RECEIPT_PAYLOAD = "d34948d26760d8beaa1d87c2dd21723bee94f06797512cbfc9944b7ed4f13637"


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
        raise RuntimeError("rsync-corrected retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong fresh parallel retry predecessor")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle deployment failed and billing was stopped"
        or receipt.get("executor_started_at") is not None
    ):
        raise RuntimeError("wrong result-blind rsync failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU private-Kaggle parallel rsync-corrected retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_deployer"] = identity(args.deployer)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_fresh_retry_operational_allowance_usd": 0.02,
            "compute_cutoff_usd": 19.98,
            "maximum_provider_wall_seconds_at_price_ceiling": 56_193,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 55_893,
        }
    )
    plan["result_blind_verified_parallel_cpu_rsync_retry"] = {
        "schema_version": "1.0",
        "status": "public rsync install correction frozen before replacement CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "executor_started": False,
            "scientific_outputs_inspected": False,
        },
        "correction": {
            "install_public_rsync_before_transfer": True,
            "private_file_transfer_started_in_failed_attempt": False,
            "private_dataset_pull_started_in_failed_attempt": False,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "held_out_result_opened_or_used": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_PARALLEL_RSYNC_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
