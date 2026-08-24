#!/usr/bin/env python3
"""Freeze the result-blind RunPod real-scoring retry and worker fan-out."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "9940f7db306e15398dcbe9ed44e270ac0d6d41e1",
    "file": "heldout_execution_plan.json",
    "bytes": 36548,
    "sha256": "3d57362fea73943f4f884a0427c286537573e669f0e999f9cc6cef29d12a02e7",
    "payload_sha256": "42cb86c2a450ddaa08e5748e94284c19b59be622f8cb6888fb32a5a6667776d8",
}
PREDECESSOR_DELIVERY = {
    "commit": "be34445c6ca3b1a82f213e0cce05a2379b30c14c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 18589,
    "sha256": "f1b94e74874c4625b60b4db2a70cc9e7bc81f5326992b538bf620d5702979dad",
    "payload_sha256": "c1e1b8482fd15cda44f7367a553c696276ea8c9db051580b8b349cb67d953287",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal characters")
    return value


def correction_record() -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind RunPod real-scoring retry frozen before private cache egress",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "failed_real_attempt": {
            "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
            "kernel_version": 6,
            "status": "ERROR",
            "failure_stage": "held-out input staging before fixed panels or scorer invocation",
            "scientific_result_created_or_opened": False,
        },
        "provider": {
            "name": "RunPod",
            "pod_id": "9r1cm9r82aonih",
            "machine_id": "zgxuf36p5h0i",
            "gpu_type": "RTX 4090",
            "gpu_count": 7,
            "vcpu_count": 224,
            "memory_gb": 411,
            "price_usd_per_hour": 2.38,
            "reuse_existing_stopped_volume": True,
        },
        "budget": {
            "absolute_cap_usd": 3.50,
            "compute_cutoff_usd": 3.25,
            "reserve_usd": 0.25,
            "guard_seconds": 300,
            "no_additional_spend_authorized": True,
        },
        "correction": {
            "official_metric_subprocesses_run_concurrently": 32,
            "per_cache_metric_function_changed": False,
            "result_rows_reassembled_in_original_frozen_cache_order": True,
            "fixed_panel_renderer_runs_before_real_scorer": True,
            "only_sealed_real_caches_and_public_metric_assets_are_newly_transferred": True,
            "preexisting_private_assets_are_reused_in_place": True,
            "aggregation_bootstrap_or_json_serialization_changed": False,
            "synthetic_v4_is_retained_without_rerun": True,
        },
        "scientific_gate": {
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "official_metric_source_or_runtime_changed": False,
            "held_out_cache_or_manifest_changed": False,
            "test_time_tuning_permitted": False,
            "failed_or_sealed_results_used_to_design_correction": False,
        },
    }


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("plan is not the exact accelerated-real predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("accelerated-real predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_runpod_real_scoring_retry"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_RUNPOD_REAL_SCORING_RETRY_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact accelerated-real predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("accelerated-real predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_runpod_real_scoring_retry") != correction_record():
        raise RuntimeError("RunPod real-scoring correction is absent")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_runpod_real_scoring_retry"] = {
        "public_execution_plan_commit": commit,
        "failed_real_version": 6,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "runpod_real_retry_permitted": True,
        "synthetic_v4_retained": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("RESULT_BLIND_RUNPOD_REAL_SCORING_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = subparsers.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--alias", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
