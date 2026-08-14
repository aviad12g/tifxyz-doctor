#!/usr/bin/env python3
"""Freeze the exact CPU-only RunPod real-scoring execution plan."""

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
    parser.add_argument("--scoring-asset-root", type=Path, required=True)
    parser.add_argument("--frozen-thresholds", type=Path, required=True)
    parser.add_argument("--embedded-real-launcher", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--orchestrator", type=Path, required=True)
    parser.add_argument("--deployer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("CPU-only RunPod real-scoring plan output must start absent")
    plan = load_hashed(args.plan)
    delivery = load_hashed(args.delivery)
    sealed = load_hashed(args.sealed_input_manifest)
    asset_manifest_path = args.scoring_asset_root / "scoring_asset_subset_manifest.json"
    asset_manifest = load_hashed(asset_manifest_path)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    public_delivery_commit = require_commit(args.public_delivery_commit, "public delivery commit")
    public_runpod_commit = require_commit(args.public_runpod_commit, "public RunPod tooling commit")
    if delivery.get("public_execution_plan") != identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("public delivery does not bind the CPU-only execution plan")
    if plan.get("result_blind_runpod_cpu_real_scoring", {}).get("provider", {}).get("gpu_count") != 0:
        raise RuntimeError("public execution plan is not CPU-only")
    if sealed.get("counts") != {
        "jobs": 7,
        "manifests": 7,
        "sealed_npz_caches": 266,
        "sealed_npz_bytes": 5_791_045_122,
    }:
        raise RuntimeError("sealed real-input aggregate mismatch")
    if asset_manifest.get("files") != 5 or asset_manifest.get("bytes") != 109_850:
        raise RuntimeError("minimal scoring-asset aggregate mismatch")
    if asset_manifest.get("scoring_ledger") != {
        "file": "SOURCE_SHA256SUMS",
        "records": 5,
        "sha256": "77b8babe8c2641aa0f1ec082c8f4fcb7c5dc808df39c2262b9f43aa359d8bff9",
    }:
        raise RuntimeError("minimal scoring-asset ledger mismatch")
    if asset_manifest.get("parent_ledger") != {
        "file": "PARENT_SOURCE_SHA256SUMS",
        "records": 278,
        "sha256": "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba",
    }:
        raise RuntimeError("parent scoring-asset ledger mismatch")
    threshold_identity = identity(args.frozen_thresholds)
    if threshold_identity["sha256"] != plan["threshold_binding"]["frozen_thresholds"]["sha256"]:
        raise RuntimeError("frozen-threshold identity mismatch")
    payload = {
        "schema_version": "1.0",
        "status": "result-blind CPU-only RunPod real one-shot scoring frozen before private cache egress",
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
        "scoring_assets": {
            "manifest": identity(asset_manifest_path) | {
                "payload_sha256": asset_manifest["payload_sha256"]
            },
            "ledger_sha256": asset_manifest["scoring_ledger"]["sha256"],
            "ledger_records": 5,
            "parent_ledger_sha256": asset_manifest["parent_ledger"]["sha256"],
            "parent_ledger_records": 278,
            "files": 5,
            "bytes": 109_850,
            "model_checkpoints_excluded": True,
        },
        "frozen_thresholds": threshold_identity,
        "embedded_real_launcher": identity(args.embedded_real_launcher),
        "input_collector": identity(args.collector),
        "remote_executor": identity(args.executor),
        "provider_orchestrator": identity(args.orchestrator),
        "deployer": identity(args.deployer),
        "provider": {
            "name": "vesuvius-real-scoring-cpu-32",
            "image": "runpod/stack",
            "compute_type": "CPU",
            "cloud_type": "SECURE",
            "cpu_flavor": "cpu3g",
            "instance_id": "cpu3g-32-128",
            "gpu_count": 0,
            "vcpu_count": 32,
            "minimum_memory_gb": 120,
            "maximum_price_usd_per_hour": 1.25,
            "container_disk_gb": 30,
            "new_temporary_pod": True,
            "temporary_public_ssh": True,
            "terminate_after_verified_result_copy": True,
        },
        "budget": {
            "absolute_cap_usd": 3.50,
            "compute_cutoff_usd": 3.20,
            "reserve_usd": 0.25,
            "prior_terminated_attempt_upper_bound_usd": 0.05,
            "guard_seconds": 300,
            "maximum_provider_wall_seconds_at_price_ceiling": 9216,
            "maximum_runtime_seconds_before_guard_at_price_ceiling": 8916,
            "on_cutoff": "stop the CPU Pod, preserve sealed volume artifacts, and never score a partial result",
        },
        "new_transfer": {
            "private": "only 266 sealed real NPZ caches plus their seven indexes and seven manifests",
            "public": "five ledger-bound scoring assets, frozen thresholds, pinned metric source/runtime, launcher, and controllers",
            "private_upload_authorized_by_aviad": True,
            "private_npz_payloads_opened_before_transfer": False,
            "new_private_checkpoint_or_research_input_upload": False,
        },
        "runtime": {
            "python": "3.12.13",
            "uv_bootstrap": "0.8.22",
            "real_parallel_workers": 32,
            "fixed_panels_before_scoring": True,
            "row_order": "original frozen cache order",
            "sealed_cache_layout_adapter": (
                "relative directory symlink from each frozen run name to its verified "
                "sealed-caches directory; no NPZ is copied or opened"
            ),
        },
        "scientific_gate": {
            "model_metric_threshold_seed_panel_endpoint_gate_or_claim_changed": False,
            "scientific_result_opened_or_used": False,
            "synthetic_version_4_remains_complete_and_sealed": True,
            "partial_real_result_is_scorable": False,
            "all_adverse_null_or_failure_outcomes_must_be_published": True,
            "gpu_compute_permitted": False,
        },
        "pre_create_events": [
            {
                "pod_id": "j0d4boyxwegn7q",
                "route": "RunPod v1 REST CPU placement",
                "result": "created allocation failed frozen provider-shape validation and was immediately terminated",
                "private_transfer_started": False,
                "scientific_outputs_inspected": False,
                "conservative_spend_upper_bound_usd": 0.05,
            }
        ],
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(args.out)
    print("RUNPOD_CPU_REAL_EXECUTION_PLAN_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
