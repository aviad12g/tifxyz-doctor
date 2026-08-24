#!/usr/bin/env python3
"""Freeze the CPU retry with bounded scorer exception diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "91e1f453fb0d5ac64b3dce95d2bf1b98cef427f1baca44aa2de8a8c59224369c"
FAILED_RECEIPT_PAYLOAD = "ff6d5dfc169e14b8159a0bf47ba9aed1c44d47c3aa5d9a16a6d7386236eef7a0"


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
        raise RuntimeError("scorer-exception retry plan target must start absent")
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
        raise RuntimeError("wrong corrected-stager predecessor plan")
    if receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD or receipt.get("scientific_outputs_inspected") is not False:
        raise RuntimeError("wrong result-blind scorer failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU bounded scorer-exception retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["embedded_real_launcher"] = identity(args.launcher)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_scorer_failure_guarded_allowance_usd": 0.4,
            "compute_cutoff_usd": 12.0,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 33_750,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 33_450,
        }
    )
    plan["result_blind_bounded_scorer_exception_retry"] = {
        "schema_version": "1.0",
        "status": "bounded scorer stderr exception diagnostic frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "private_transport_verified": True,
            "credentials_removed": True,
            "scorer_returned_nonzero": True,
            "scientific_outputs_inspected": False,
        },
        "scientific_contract": {
            "only_bounded_stderr_exception_projection_changed": True,
            "scientific_stdout_projected": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_BOUNDED_SCORER_EXCEPTION_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
