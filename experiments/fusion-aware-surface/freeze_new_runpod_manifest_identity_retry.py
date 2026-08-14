#!/usr/bin/env python3
"""Freeze the exact private transport manifest-identity correction."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "71ab4b8171dd1f91bf00cebcdb53e6abd6b73e05726478b5323e8ee00e8afb39"
FAILED_RECEIPT_PAYLOAD = "3ff42e51ae0be7015baec33d340d3d68b393dff4eb127f65345add1d29110aea"
MANIFEST_SHA256 = "c8da09eb8477f4f00577cca054737c73b6c9d6d0fb5758ec683b2e1f7c14ec7c"


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
        raise RuntimeError("manifest-identity retry plan target must start absent")
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
        raise RuntimeError("wrong access-token-corrected predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle verified-parallel pipeline started"
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind manifest-identity failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU exact manifest-file-identity retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["private_transport_puller"] = identity(args.puller)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_manifest_identity_retry_operational_allowance_usd": 0.5,
            "compute_cutoff_usd": 18.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 50_625,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 50_325,
        }
    )
    plan["result_blind_transport_manifest_file_identity_retry"] = {
        "schema_version": "1.0",
        "status": "exact three-field manifest file-identity correction frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "failure_state": "PULLING_PRIVATE_TRANSPORT",
            "failure_message": "private transport staging-manifest identity mismatch",
            "credentials_removed": True,
            "real_scorer_started": False,
            "scientific_outputs_inspected": False,
        },
        "independent_metadata_check": {
            "remote_manifest_bytes": 1078,
            "remote_manifest_sha256": MANIFEST_SHA256,
            "local_manifest_bytes": 1078,
            "local_manifest_sha256": MANIFEST_SHA256,
            "npz_downloaded_opened_or_parsed": False,
        },
        "correction": {
            "file_identity_fields": ["file", "bytes", "sha256"],
            "payload_sha256_validated_separately": True,
            "manifest_bytes_changed": False,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "private_dataset_id_version_or_files_changed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_MANIFEST_FILE_IDENTITY_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
