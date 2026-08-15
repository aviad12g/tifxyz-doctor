#!/usr/bin/env python3
"""Freeze ephemeral signed-bundle transport for a fresh CPU scorer retry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "387afb1423b8f0f85027b387bff80fd7d54586f87a4bd110ed7e11512309ec01"
FAILED_RECEIPT_PAYLOAD = "d3ef199d6e5c8d85708172d7542e2d0c73218af351bebb4e0ba23d0315202b7f"


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
        raise RuntimeError("signed-bundle retry plan target must start absent")
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
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong in-memory-auth predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("scientific_outputs_inspected") is not False
        or receipt.get("pod_id") != "nj7n06gzpklal8"
    ):
        raise RuntimeError("wrong result-blind remote unauthenticated receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU ephemeral signed-bundle retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_puller"] = identity(args.puller)
    plan["private_transport_wrapper"] = identity(args.wrapper)
    plan["private_transport_deployer"] = identity(args.deployer)
    plan["fresh_pod_deployer"] = identity(args.deployer)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_remote_unauthenticated_guarded_allowance_usd": 0.5,
            "compute_cutoff_usd": 9.5,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 26_718,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 26_418,
        }
    )
    plan["result_blind_ephemeral_signed_bundle_retry"] = {
        "schema_version": "1.0",
        "status": "ephemeral signed private-bundle transport frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "failure_state": "PULLING_PRIVATE_TRANSPORT",
            "failure_type": "UnauthenticatedError",
            "credentials_removed": True,
            "private_cache_download_started": False,
            "real_scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "signed_transport": {
            "long_lived_kaggle_credential_leaves_local_machine": False,
            "signed_url_value_or_hash_frozen_logged_or_recorded": False,
            "signed_url_expected_expiry_seconds": 259_200,
            "signed_bundle_expected_bytes": 5_791_073_068,
            "remote_preflight_range_bytes": "0-0",
            "remote_preflight_before_large_public_transfer": True,
            "signed_url_file_removed_before_bundle_request": True,
            "archive_members_safety_checked_before_extraction": True,
            "downloaded_files_rehashed_without_npz_parsing": True,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "private_dataset_id_or_version_changed": False,
            "private_dataset_files_changed": False,
            "npz_payloads_opened_or_parsed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_EPHEMERAL_SIGNED_BUNDLE_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
