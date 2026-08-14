#!/usr/bin/env python3
"""Freeze the exact original scorer path after CPU projection preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "99643d5ed891611ac333fc222c1040c37f8c2a3b",
    "file": "heldout_execution_plan.json",
    "bytes": 44210,
    "sha256": "f1f4544a4b94cf5ba388d80d5dca441d1704db1d00821dbd6da899d140ab3f1f",
    "payload_sha256": "dde5f0a0d566f02f9f6623a3528f9727154330a0d7ebc77684e078ff82ed6056",
}
PREDECESSOR_DELIVERY = {
    "commit": "4b5baaf5e23af0b8efbe74b90cd17157c695f10c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 19598,
    "sha256": "eebf95021f4d5d305298ea52e0b3059e1c5dc877d1dafaa8de99edc5574bab29",
    "payload_sha256": "2c23316d23d087b1dcf9d95815bcb6c1097a8b228984465592c090a21f7810b0",
}
ORIGINAL_REAL_SCORER = {
    "file": "score_real_test.py",
    "sha256": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
}
PARALLEL_REAL_SCORER = {
    "file": "score_real_test.py",
    "sha256": "06119047c636185f70455feac6ed780510dcd4b7f636404abfdbad819cae1b16",
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


def require_commit(value: str, label: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def correction_record(public_tooling_commit: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind restoration of exact original real scorer for CPU execution",
        "public_tooling_commit": require_commit(public_tooling_commit, "public tooling commit"),
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "local_preflight": {
            "provider_attempt_created": False,
            "remaining_differences": [
                "CPU scoring-asset stager identity is operational metadata",
                "parallel real scorer identity differs from the cache-job scientific plan",
            ],
        },
        "decision": {
            "parallel_scorer_scientific_equivalence_assumed": False,
            "parallel_scorer_used_for_retry": False,
            "exact_original_real_scorer_restored": ORIGINAL_REAL_SCORER,
            "CPU_launcher_passes_no_parallel_worker_arguments": True,
            "fixed_panel_renderer_still_runs_before_the_scorer": True,
        },
        "scientific_gate": {
            "cache_npz_opened_or_inspected": False,
            "probability_or_endpoint_opened": False,
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "test_time_tuning_permitted": False,
            "scientific_result_created_or_opened": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("execution plan is not the exact CPU exact-scorer predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("predecessor execution-plan payload mismatch")
    if plan.get("one_shot_scorers", {}).get("real", {}).get("script") != PARALLEL_REAL_SCORER:
        raise RuntimeError("predecessor parallel real-scorer identity mismatch")
    plan["one_shot_scorers"]["real"]["script"] = ORIGINAL_REAL_SCORER
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_runpod_cpu_exact_scorer_correction"] = correction_record(
        args.public_tooling_commit
    )
    write_hashed(args.plan, plan)
    print("RUNPOD_CPU_EXACT_SCORER_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        key: PREDECESSOR_DELIVERY[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("delivery is not the exact CPU exact-scorer predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("predecessor cache-delivery payload mismatch")
    plan = load_hashed(args.plan)
    record = plan.get("result_blind_runpod_cpu_exact_scorer_correction")
    if record != correction_record(args.public_tooling_commit):
        raise RuntimeError("exact-scorer correction is absent or changed")
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_runpod_cpu_exact_scorer_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": record["public_tooling_commit"],
        "exact_original_real_scorer_restored": True,
        "scientific_contract_changed": False,
        "scientific_result_created_or_opened": False,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("RUNPOD_CPU_EXACT_SCORER_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
    plan.add_argument("--public-tooling-commit", required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = subparsers.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--alias", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-tooling-commit", required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
