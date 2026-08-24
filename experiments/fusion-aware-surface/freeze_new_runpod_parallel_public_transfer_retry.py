#!/usr/bin/env python3
"""Freeze the bounded parallel public-runtime transfer CPU retry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "40d020d8fdff70477806bea16d745f9c4a7289e91b47e7f6d1b49f4b5a6ad314"
FAILED_RECEIPT_PAYLOAD = "510eded976df38ec5734dfbf679a6e3f677fb81678792a31e3d4dc1cf269e7a1"


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
        raise RuntimeError("parallel-transfer retry plan target must start absent")
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
        raise RuntimeError("wrong launcher-diagnostic predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle deployment failed and billing was stopped"
        or receipt.get("error_type") != "TimeoutExpired"
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind public-transfer timeout receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU parallel public-runtime transfer retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["fresh_pod_deployer"] = identity(args.deployer)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_public_transfer_timeout_guarded_allowance_usd": 0.5,
            "compute_cutoff_usd": 14.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 39_375,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 39_075,
        }
    )
    plan["result_blind_parallel_public_runtime_transfer_retry"] = {
        "schema_version": "1.0",
        "status": "bounded parallel public-runtime transfer frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "private_transport_started": False,
            "scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "correction": {
            "public_metric_runtime_files": 20,
            "maximum_parallel_ssh_streams": 8,
            "metric_runtime_tree_identity_changed": False,
            "remote_wrapper_rehashes_complete_tree_before_private_pull": True,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
            "parallel_scorer_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_PARALLEL_PUBLIC_TRANSFER_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
