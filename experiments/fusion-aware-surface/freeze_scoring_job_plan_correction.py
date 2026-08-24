#!/usr/bin/env python3
"""Freeze the result-blind cache-job-plan compatibility correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "393dee840bf4ef6c75f228023d089a8ea22c8785",
    "file": "heldout_execution_plan.json",
    "bytes": 22974,
    "sha256": "c01a9dc7274c638f1206eea1a68cf91721464b5ffc04acc319e93fe98d393b70",
    "payload_sha256": "68d649f1c97484d0ec9974763e60b64ca57004661dcf3d55806a86d7f112361c",
}
PREDECESSOR_DELIVERY = {
    "commit": "3c8722ccf6d13b046367496af785a99a7ce1c5af",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 15425,
    "sha256": "8f9dff27af3c1175edc40ae8cbd1dfc2678d9830c9e47f06b7bda7666c616f23",
    "payload_sha256": "8bf9473e690d25a8bb1462b0d1ac6afea7293e2cca317b07a39f6c3fabc60953",
}
CACHE_JOB_PLAN = {
    "commit": "46d030245c14ecb2be63796a308581c462fc2e29",
    "file": "heldout_execution_plan.json",
    "bytes": 20443,
    "sha256": "29d08a7430c8870582a96579fcc77737ac699aef64efc159ba94ede87009da5d",
    "payload_sha256": "3826bf641941b5d70fa52f9a23fc1711ae538be8fc974dbba1b5f0db8f18afdd",
}
FAILED_ATTEMPTS = [
    {
        "mode": "real",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
        "kernel_version": 2,
    },
    {
        "mode": "synthetic",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
        "kernel_version": 2,
    },
]


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
        "status": "result-blind cache-job-plan compatibility correction after pre-scoring rejection",
        "predecessor_corrected_execution_plan": PREDECESSOR_PLAN,
        "predecessor_corrected_cache_delivery": PREDECESSOR_DELIVERY,
        "cache_job_execution_plan": CACHE_JOB_PLAN,
        "failed_attempts": FAILED_ATTEMPTS,
        "failure_stage": "held-out input staging before scorer invocation",
        "cause": (
            "immutable cache-job indexes correctly bind the original pre-inference plan; "
            "the corrected scorer stager incorrectly required them to bind the later "
            "operational-only scoring plan"
        ),
        "correction": {
            "current_scoring_plan_remains_publicly_hash_bound": True,
            "cache_job_indexes_verified_against_original_plan": True,
            "scientific_projection_of_both_plans_must_match_exactly": True,
            "cache_job_index_or_manifest_changed": False,
        },
        "scientific_gate": {
            "input_staging_process_started_in_failed_attempts": True,
            "cache_job_index_metadata_may_have_been_read": True,
            "cache_npz_opened_or_inspected": False,
            "scorer_invocation_started": False,
            "scientific_result_created_or_opened": False,
            "threshold_seed_panel_gate_or_claim_changed": False,
            "retry_is_first_scientific_scorer_invocation": True,
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
        raise RuntimeError("execution plan is not the exact corrected predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("corrected predecessor plan payload mismatch")
    first = plan.get("result_blind_scoring_asset_correction", {})
    if first.get("predecessor_public_execution_plan") != CACHE_JOB_PLAN:
        raise RuntimeError("original cache-job plan identity is absent")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_scoring_job_plan_compatibility_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_CACHE_JOB_PLAN_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("cache delivery is not the exact corrected predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("corrected predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_scoring_job_plan_compatibility_correction") != correction_record():
        raise RuntimeError("cache-job plan correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scoring_job_plan_compatibility_correction"] = {
        "public_execution_plan_commit": commit,
        "cache_job_execution_plan": CACHE_JOB_PLAN,
        "failed_attempts": FAILED_ATTEMPTS,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "retry_permitted": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery filename alias differs byte-for-byte")
    print("RESULT_BLIND_CACHE_JOB_DELIVERY_CORRECTED")


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
