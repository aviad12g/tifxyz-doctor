#!/usr/bin/env python3
"""Freeze direct in-memory legacy Kaggle authentication for a fresh CPU retry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "5932db7daf8900430b2285765dbc058654c2c2c35d383960bee07b01d72b03a3"
FAILED_RECEIPT_PAYLOAD = "d400ecfd6826fb5c90ef11be65dd4b3cbf351bb0096e0e2b95a96b873b0b8c59"


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
        raise RuntimeError("in-memory-credential retry plan target must start absent")
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
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong legacy-JSON predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("scientific_outputs_inspected") is not False
        or receipt.get("pod_id") != "mwmfigmy78uy86"
    ):
        raise RuntimeError("wrong result-blind private-auth failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU in-memory legacy-Kaggle retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_puller"] = identity(args.puller)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_in_memory_auth_failure_guarded_allowance_usd": 0.5,
            "compute_cutoff_usd": 10.5,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 29_531,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 29_231,
        }
    )
    plan["result_blind_in_memory_kaggle_credentials_retry"] = {
        "schema_version": "1.0",
        "status": "direct in-memory legacy Kaggle authentication frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "failure_state": "PULLING_PRIVATE_TRANSPORT",
            "failure_status_code": 403,
            "credentials_removed": True,
            "private_cache_download_started": False,
            "real_scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "correction": {
            "credential_kind": "legacy Kaggle username/key JSON",
            "authentication_method": "KaggleHub 1.0.2 in-memory set_kaggle_credentials",
            "potential_override_environment_variables_cleared": [
                "KAGGLE_API_TOKEN",
                "KAGGLE_USERNAME",
                "KAGGLE_KEY"
            ],
            "authenticated_owner_must_equal_frozen_dataset_owner": True,
            "credential_file_removed_before_dataset_request": True,
            "credential_value_or_hash_frozen_logged_or_recorded": False,
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
    print("NEW_RUNPOD_IN_MEMORY_KAGGLE_CREDENTIALS_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
