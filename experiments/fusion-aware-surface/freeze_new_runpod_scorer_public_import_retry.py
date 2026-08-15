#!/usr/bin/env python3
"""Freeze the CPU retry with exact public scorer import dependencies."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "6ae28cd20f7a64c34aca81d234ce188b11fe9b423b144f76fd52ee2727123869"
FAILED_RECEIPT_PAYLOAD = "9c347768feeef2123b49f33f5db6e86920b5fbcfc2435d18465b6ca262c8e2b6"
PUBLIC_IMPORTS = (
    "fusion_loss.py",
    "gap_supervision.py",
    "inference.py",
    "normalization.py",
    "train_fusion_aware.py",
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
    if (
        not isinstance(payload, dict)
        or payload.get("payload_sha256") != canonical_sha256(payload)
    ):
        raise RuntimeError(f"invalid hashed JSON: {path}")
    return payload


def identity(path: Path) -> dict:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_commit(value: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            "public tooling commit must be 40 lowercase hexadecimal characters"
        )
    return value


def write_hashed(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError("public-import retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong signed-bundle predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("scientific_outputs_inspected") is not False
        or receipt.get("pod_id") != "djai7jtnk3helh"
    ):
        raise RuntimeError("wrong result-blind missing-public-import receipt")
    plan = deepcopy(predecessor)
    plan["status"] = (
        "fresh CPU exact public scorer-import dependency retry frozen before create"
    )
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["embedded_real_launcher"] = identity(args.launcher)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_public_import_failure_guarded_allowance_usd": 0.4,
            "compute_cutoff_usd": 9.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 25_312,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 25_012,
        }
    )
    plan["result_blind_scorer_public_import_dependency_retry"] = {
        "schema_version": "1.0",
        "status": "exact hash-bound public scorer imports frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "private_signed_bundle_transport_verified": True,
            "failure_type": "ModuleNotFoundError",
            "failure_message": "No module named 'inference'",
            "scorer_returned_nonzero": True,
            "scientific_outputs_inspected": False,
        },
        "public_import_dependencies": [
            identity(args.source_root / name) for name in PUBLIC_IMPORTS
        ],
        "scientific_contract": {
            "only_exact_existing_public_import_dependencies_materialized": True,
            "dependency_source_bytes_changed": False,
            "private_transport_or_cache_bytes_changed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_SCORER_PUBLIC_IMPORT_DEPENDENCY_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
