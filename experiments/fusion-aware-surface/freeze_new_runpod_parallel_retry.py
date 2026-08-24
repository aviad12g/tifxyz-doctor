#!/usr/bin/env python3
"""Freeze a fresh CPU-only private-Kaggle verified-parallel retry."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


PREDECESSOR_PAYLOAD = "947df4e646fb5a5925609d176773e3323bcad690ef937ebbf4bbd589bb88247b"
PARALLEL_EQUIVALENCE_PAYLOAD = "d373745060c54a956191af64f0fd49fdea75aaade60ee61abe11cf555ac64e67"


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


def public_identity(path: Path, commit: str) -> dict:
    payload = load_hashed(path)
    return identity(path) | {"commit": commit, "payload_sha256": payload["payload_sha256"]}


def write_hashed(path: Path, payload: dict) -> None:
    if path.exists():
        raise RuntimeError("fresh parallel retry plan target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--public-tooling-commit", required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--execution-plan-commit", required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--delivery-commit", required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    predecessor = load_hashed(args.predecessor)
    if predecessor["payload_sha256"] != PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong private-Kaggle predecessor plan")
    tooling_commit = require_commit(args.public_tooling_commit)
    execution_commit = require_commit(args.execution_plan_commit)
    delivery_commit = require_commit(args.delivery_commit)
    report = load_hashed(args.equivalence_report)
    if report["payload_sha256"] != PARALLEL_EQUIVALENCE_PAYLOAD:
        raise RuntimeError("wrong parallel equivalence report")
    plan = deepcopy(predecessor)
    plan["status"] = "fresh CPU-only private-Kaggle verified-parallel retry frozen before provider create"
    plan["public_runpod_commit"] = tooling_commit
    plan["public_execution_plan"] = public_identity(args.execution_plan, execution_commit)
    plan["public_cache_delivery"] = public_identity(args.delivery, delivery_commit)
    plan["embedded_real_launcher"] = identity(args.launcher)
    plan["private_transport_deployer"] = identity(args.deployer)
    plan["budget"] = {
        "absolute_cap_usd": 21.0,
        "additional_authorized_cap_usd": 21.0,
        "legacy_campaign_guarded_spend_upper_bound_usd": 11.65498976211111,
        "compute_cutoff_usd": 20.0,
        "reserve_usd": 1.0,
        "guard_seconds": 300,
        "maximum_provider_wall_seconds_at_price_ceiling": 56_250,
        "maximum_runtime_seconds_before_guard_at_price_ceiling": 55_950,
        "on_cutoff": "stop the CPU Pod, preserve sealed volume artifacts, and never score a partial result",
    }
    plan["result_blind_verified_parallel_cpu_retry"] = {
        "schema_version": "1.0",
        "status": "fresh CPU private-Kaggle verified-parallel retry frozen before provider create",
        "fresh_cpu_pod": True,
        "parallel_workers": 32,
        "equivalence_report": identity(args.equivalence_report) | {
            "commit": "85b0190613ea35b73f16079a67ae52d9eaa9cf8e",
            "payload_sha256": report["payload_sha256"],
        },
        "provider_contract": {
            "compute_type": "CPU",
            "gpu_count": 0,
            "vcpu_count": 32,
            "minimum_memory_gb": 120,
            "maximum_price_usd_per_hour": 1.28,
            "new_temporary_pod": True,
            "terminate_after_verified_result_copy": True,
        },
        "transport_contract": {
            "private_dataset_version": 1,
            "downloaded_file_identities_reverified": True,
            "ephemeral_credential_removed_after_download": True,
            "direct_local_sealed_cache_upload": False,
        },
        "scientific_contract": {
            "model_metric_threshold_seed_panel_endpoint_gate_aggregation_or_claim_changed": False,
            "per_cache_metric_subprocess_changed": False,
            "frozen_cache_order_preserved": True,
            "only_independent_subprocess_scheduling_changed": True,
            "held_out_result_opened_or_used": False,
            "test_time_tuning_permitted": False,
        },
    }
    if (
        plan["provider"]["compute_type"] != "CPU"
        or plan["provider"]["gpu_count"] != 0
        or plan["private_kaggle_transport"]["dataset_version"] != 1
    ):
        raise RuntimeError("predecessor provider or private transport contract changed")
    write_hashed(args.out, plan)
    print("NEW_RUNPOD_PRIVATE_KAGGLE_PARALLEL_RETRY_FROZEN")
    print(load_hashed(args.out)["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
