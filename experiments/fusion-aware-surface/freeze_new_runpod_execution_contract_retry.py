#!/usr/bin/env python3
"""Freeze the corrected CPU executor contract handoff."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "fce53b88657770c497cfbb009f6ff437c96a7cf60630d874ebd54efc5ff3ac93"
FAILED_RECEIPT_PAYLOAD = "a6600156531335198840c1b3ef1f714c1691df1ddd05b86f4cad93a900c8c97d"


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
        raise RuntimeError("executor-contract retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--failed-receipt", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    receipt = load_hashed(args.failed_receipt)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong manifest-identity-corrected predecessor plan")
    if (
        receipt["payload_sha256"] != FAILED_RECEIPT_PAYLOAD
        or receipt.get("status") != "fresh CPU private-Kaggle verified-parallel pipeline started"
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("wrong result-blind executor-contract failure receipt")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU exact execution-contract retry frozen before create"
    plan["public_runpod_commit"] = require_commit(args.public_tooling_commit)
    plan["remote_executor"] = identity(args.executor)
    plan["private_transport_wrapper"] = identity(args.wrapper)
    plan["budget"] = dict(plan["budget"])
    plan["budget"].update(
        {
            "prior_executor_contract_retry_operational_allowance_usd": 0.5,
            "compute_cutoff_usd": 17.5,
            "reserve_usd": 1.0,
            "maximum_provider_wall_seconds_at_price_ceiling": 49_218,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 48_918,
        }
    )
    plan["result_blind_corrected_executor_contract_retry"] = {
        "schema_version": "1.0",
        "status": "corrected executor contract handoff frozen before CPU create",
        "failed_receipt": {
            "file": args.failed_receipt.name,
            "payload_sha256": receipt["payload_sha256"],
            "pod_id": receipt["pod_id"],
            "private_transport_verified": True,
            "credentials_removed": True,
            "executor_returned_nonzero": True,
            "stopped_workspace_status_preserved": False,
            "scientific_outputs_inspected": False,
        },
        "deterministic_preflight": {
            "old_literal_original_status_required": True,
            "corrected_retry_status_differs": True,
            "old_executor_guaranteed_to_reject_before_science": True,
        },
        "correction": {
            "top_level_status_text_is_scientific_gate": False,
            "exact_cpu_provider_contract_required": True,
            "exact_result_blind_scientific_gate_required": True,
            "exact_32_worker_equivalence_contract_required": True,
            "all_existing_file_and_runtime_identity_checks_retained": True,
            "propagate_result_blind_executor_error": True,
            "fresh_replacement_cpu_pod": True,
        },
        "scientific_contract": {
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
            "parallel_workers": 32,
            "gpu_count": 0,
        },
    }
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_CORRECTED_EXECUTION_CONTRACT_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
