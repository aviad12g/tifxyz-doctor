#!/usr/bin/env python3
"""Freeze the exact RunPod real-scoring transport and execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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


def require_commit(value: str, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-delivery-commit", required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--public-runpod-commit", required=True)
    parser.add_argument("--sealed-input-manifest", type=Path, required=True)
    parser.add_argument("--embedded-real-launcher", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--orchestrator", type=Path, required=True)
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("RunPod real execution plan output must start absent")
    plan = load_hashed(args.plan)
    delivery = load_hashed(args.delivery)
    sealed = load_hashed(args.sealed_input_manifest)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    public_delivery_commit = require_commit(args.public_delivery_commit, "public delivery commit")
    public_runpod_commit = require_commit(args.public_runpod_commit, "public RunPod tooling commit")
    if delivery.get("public_execution_plan") != identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("public delivery does not bind the RunPod retry plan")
    if sealed.get("counts") != {
        "jobs": 7,
        "manifests": 7,
        "sealed_npz_caches": 266,
        "sealed_npz_bytes": 5_791_045_122,
    }:
        raise RuntimeError("sealed real-input aggregate mismatch")
    payload = {
        "schema_version": "1.0",
        "status": "result-blind RunPod real one-shot scoring frozen before private cache egress",
        "public_execution_plan": identity(args.plan) | {
            "commit": public_plan_commit,
            "payload_sha256": plan["payload_sha256"],
        },
        "public_cache_delivery": identity(args.delivery) | {
            "commit": public_delivery_commit,
            "payload_sha256": delivery["payload_sha256"],
        },
        "public_runpod_commit": public_runpod_commit,
        "sealed_real_inputs": identity(args.sealed_input_manifest) | {
            "payload_sha256": sealed["payload_sha256"]
        },
        "embedded_real_launcher": identity(args.embedded_real_launcher),
        "input_collector": identity(args.collector),
        "remote_executor": identity(args.executor),
        "provider_orchestrator": identity(args.orchestrator),
        "deployer": identity(args.deployer),
        "provider": {
            "id": "9r1cm9r82aonih",
            "machine_id": "zgxuf36p5h0i",
            "gpu_count": 7,
            "vcpu_count": 224,
            "memory_gb": 411,
            "price_usd_per_hour": 2.38,
            "desired_pre_resume_status": "EXITED",
            "reuse_existing_stopped_volume": True,
            "result_blind_capacity_fallback": {
                "allowed_only_after_primary_capacity_rejection": True,
                "gpu_count": 6,
                "minimum_vcpu_count": 192,
                "minimum_memory_gb": 300,
                "maximum_price_usd_per_hour": 2.04,
                "same_pod_and_machine_required": True,
            },
        },
        "budget": {
            "absolute_cap_usd": 3.50,
            "compute_cutoff_usd": 3.25,
            "reserve_usd": 0.25,
            "guard_seconds": 300,
            "maximum_guarded_wall_seconds": 4915,
            "maximum_un_guarded_wall_seconds": 4615,
            "on_cutoff": "stop the allocation, preserve sealed artifacts, and never score a partial result",
        },
        "reused_private_assets": {
            "pod_id": "9r1cm9r82aonih",
            "asset_ledger": "/workspace/bundle/input/assets/SOURCE_SHA256SUMS",
            "asset_ledger_sha256": "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba",
            "frozen_thresholds": "/workspace/bundle/input/threshold-freeze/frozen_thresholds.json",
            "frozen_thresholds_sha256": plan["threshold_binding"]["frozen_thresholds"]["sha256"],
            "new_private_checkpoint_or_research_input_upload": False,
        },
        "new_transfer": {
            "private": "only 266 sealed real NPZ caches plus their seven indexes and seven manifests",
            "public": "pinned metric source, CPython 3.12 metric wheelhouse, launcher, and controllers",
            "private_upload_authorized_by_aviad": True,
            "private_npz_payloads_opened_before_transfer": False,
        },
        "runtime": {
            "python": "3.12.13",
            "uv_bootstrap": "0.8.22",
            "real_parallel_workers": 32,
            "fixed_panels_before_scoring": True,
            "row_order": "original frozen cache order",
            "sealed_cache_layout_adapter": (
                "relative directory symlink from each frozen run name to its "
                "verified sealed-caches directory; no NPZ is copied or opened"
            ),
        },
        "scientific_gate": {
            "model_metric_threshold_seed_panel_endpoint_gate_or_claim_changed": False,
            "scientific_result_opened_or_used": False,
            "synthetic_version_4_remains_complete_and_sealed": True,
            "partial_real_result_is_scorable": False,
            "all_adverse_null_or_failure_outcomes_must_be_published": True,
        },
        "pre_resume_capacity_event": {
            "attempted_gpu_count": 7,
            "provider_result": "rejected because the original host lacked seven free GPUs",
            "billing_started": False,
            "private_transfer_started": False,
            "scientific_outputs_inspected": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(args.out)
    print("RUNPOD_REAL_EXECUTION_PLAN_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
