#!/usr/bin/env python3
"""Freeze the result-blind CPU-only RunPod real-scoring correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "file": "heldout_execution_plan.json",
    "bytes": 39063,
    "sha256": "1d67f9f87c17ca93cea612b2586ebfba4687a3e8ecbe9ed6f300a54d3b847494",
    "payload_sha256": "b600eb61f46dfeacf0b0ba12a37ecf77e553ce61d88e5a83f6056ab724337740",
}
PREDECESSOR_DELIVERY = {
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 18917,
    "sha256": "fc594e97f5d3a816cac4faac11fab545684ccc061b68a9c85f5a8bfe940303d6",
    "payload_sha256": "a22027c2068d40a6be83b63d925016600974c79264db3d60d844bed8bccbf70d",
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


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal characters")
    return value


def correction_record() -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind CPU-only RunPod real-scoring correction frozen before private cache egress",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "sealed_predecessors": {
            "synthetic_version_4_complete_and_unopened": True,
            "real_kaggle_version_6_error_and_unopened": True,
            "failed_gpu_resume_attempts_billed_or_transferred_private_data": False,
            "scientific_outputs_used_to_design_correction": False,
        },
        "provider": {
            "name": "RunPod",
            "compute_type": "CPU",
            "gpu_count": 0,
            "cpu_flavor": "cpu3g",
            "vcpu_count": 32,
            "minimum_memory_gb": 120,
            "maximum_price_usd_per_hour": 1.25,
            "new_temporary_pod": True,
            "temporary_public_ssh": True,
            "terminate_after_verified_result_copy": True,
        },
        "budget": {
            "absolute_cap_usd": 3.50,
            "compute_cutoff_usd": 3.25,
            "reserve_usd": 0.25,
            "guard_seconds": 300,
            "user_authorized_new_cpu_pod_and_private_upload": True,
        },
        "transfer": {
            "private": "266 sealed real NPZ caches, seven indexes, and seven manifests",
            "public": "five scoring-only source assets, frozen thresholds, pinned metric source/runtime, and controllers",
            "new_private_checkpoint_or_research_input_upload": False,
            "model_checkpoints_excluded": True,
            "private_npz_payloads_opened_before_transfer": False,
        },
        "correction": {
            "official_metric_subprocesses_run_concurrently": 32,
            "per_cache_metric_function_changed": False,
            "result_rows_reassembled_in_original_frozen_cache_order": True,
            "fixed_panel_renderer_runs_before_real_scorer": True,
            "minimal_asset_subset_is_byte_bound_to_original_278_record_ledger": True,
            "aggregation_bootstrap_or_json_serialization_changed": False,
            "synthetic_v4_is_retained_without_rerun": True,
            "sealed_cache_layout_adapter": (
                "relative directory symlink from each frozen run name to its verified "
                "sealed-caches directory; no NPZ is copied or opened"
            ),
        },
        "scientific_gate": {
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "official_metric_source_or_runtime_changed": False,
            "held_out_cache_or_manifest_changed": False,
            "test_time_tuning_permitted": False,
            "failed_or_sealed_results_used_to_design_correction": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("execution plan is not the exact CPU-correction predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("predecessor execution-plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["runpod_cpu_scoring_asset_stager"] = identity(args.asset_stager)
    plan["result_blind_runpod_cpu_real_scoring"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_RUNPOD_CPU_REAL_SCORING_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        key: PREDECESSOR_DELIVERY[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("cache delivery is not the exact CPU-correction predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("predecessor cache-delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_runpod_cpu_real_scoring") != correction_record():
        raise RuntimeError("CPU-only RunPod correction is absent")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_runpod_cpu_real_scoring"] = {
        "public_execution_plan_commit": commit,
        "cpu_only_retry_permitted": True,
        "gpu_compute_permitted": False,
        "scientific_contract_changed": False,
        "scientific_result_created_or_opened": False,
        "synthetic_v4_retained": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("RESULT_BLIND_RUNPOD_CPU_REAL_SCORING_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--asset-stager", type=Path, required=True)
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
