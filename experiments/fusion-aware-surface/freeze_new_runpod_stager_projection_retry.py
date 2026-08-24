#!/usr/bin/env python3
"""Freeze the CPU retry after corrected scorer-projection publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "65d41029d4d0a1eb1b4cceba9926929cf2e500876f0bae47174e414b116c2d58"
FAILED_RECEIPT_PAYLOAD = "190ee66c4c9e7172fdf608ce058e138028ba07813ae36def747e699e59303365"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    content.pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        raise RuntimeError("stager-projection retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong parallel-public-transfer predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind stager-projection failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU corrected stager-projection retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["embedded_real_launcher"] = identity(args.launcher)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_stager_projection_failure_guarded_allowance_usd": 0.4,
            "compute_cutoff_usd": 13.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 36_562,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 36_262,
        }
    )
    plan["result_blind_corrected_stager_projection_retry"] = {
        "schema_version": "1.0",
        "status": "corrected verified-parallel scorer projection frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "private_transport_verified": True,
            "credentials_removed": True,
            "panels_or_scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "scientific_contract": {
            "only_verified_parallel_scorer_identity_projection_changed": True,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_CORRECTED_STAGER_PROJECTION_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
