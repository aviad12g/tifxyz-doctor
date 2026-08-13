#!/usr/bin/env python3
"""Freeze the result-blind parallel real-scoring and panel fail-fast retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "624c1ef41fa078ea9b3d3d97207a56ab6662e27b",
    "file": "heldout_execution_plan.json",
    "bytes": 32213,
    "sha256": "e1a2741bfae98eb5054c3c4a11195fa6ee265b3aa7848eb8117a68ac45ad09bd",
    "payload_sha256": "d60199310fd589cef799a66b3a4513927aeba9edae1914c2f54da25384d383cf",
}
PREDECESSOR_DELIVERY = {
    "commit": "367050ad1ff23f97be7996fffb978220678bd91c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 17685,
    "sha256": "a249eb36c68ea4fe13b34c549adaf42c91754e3a45c24de1dff179aeb83189f0",
    "payload_sha256": "e76019aaba37574cbc8899ede3cc939f065246d3012ac2d4efd70d3c55571afd",
}
REAL_V4 = {
    "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
    "kernel_version": 4,
    "status": "ERROR",
}
SYNTHETIC_V4 = {
    "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
    "kernel_version": 4,
    "status": "COMPLETE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def script_identity(path: Path) -> dict:
    return {"file": path.name, "sha256": sha256_file(path)}


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
        "status": "result-blind parallel real-scoring and panel fail-fast retry",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "provider_attempts": {"real": REAL_V4, "synthetic": SYNTHETIC_V4},
        "failure_stage": "after real scorer completion, during fixed real-panel rendering",
        "failure_message": "fixed real-panel rendering failed; output remains sealed",
        "failure_elapsed_seconds": 34706,
        "sealed_state": {
            "real_v4_scorer_invocation_completed": True,
            "real_v4_kernel_complete_artifact_set_available": False,
            "real_v4_result_opened_downloaded_or_used": False,
            "synthetic_v4_result_opened_downloaded_or_used": False,
            "cache_npz_probability_endpoint_or_panel_opened": False,
        },
        "correction": {
            "fixed_panel_renderer_runs_before_long_real_scorer": True,
            "panel_failure_stderr_may_be_emitted_as_operational_log": True,
            "official_metric_subprocesses_run_concurrently": 4,
            "per_cache_metric_function_changed": False,
            "result_rows_reassembled_in_original_frozen_cache_order": True,
            "aggregation_bootstrap_or_json_serialization_changed": False,
            "synthetic_v4_is_retained_without_rerun": True,
        },
        "scientific_gate": {
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "official_metric_source_or_runtime_changed": False,
            "held_out_cache_or_manifest_changed": False,
            "test_time_tuning_permitted": False,
            "sealed_v4_results_used_to_design_correction": False,
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
        raise RuntimeError("execution plan is not the exact v4 predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("v4 predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scorers"]["real"]["script"] = script_identity(args.real_scorer)
    plan["result_blind_parallel_real_scoring_retry"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_PARALLEL_REAL_SCORING_RETRY_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact v4 predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("v4 predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_parallel_real_scoring_retry") != correction_record():
        raise RuntimeError("parallel real-scoring retry correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_parallel_real_scoring_retry"] = {
        "public_execution_plan_commit": commit,
        "real_v4": REAL_V4,
        "synthetic_v4": SYNTHETIC_V4,
        "sealed_v4_results_opened_downloaded_or_used": False,
        "scientific_contract_changed": False,
        "real_v5_retry_permitted": True,
        "synthetic_v4_retained": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery filename alias differs byte-for-byte")
    print("RESULT_BLIND_PARALLEL_REAL_SCORING_RETRY_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--real-scorer", type=Path, required=True)
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
