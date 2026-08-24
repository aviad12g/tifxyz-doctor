#!/usr/bin/env python3
"""Freeze the result-blind cache-identity schema staging correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "52d73f8fc3125e9c8ec471000795192011aee954",
    "file": "heldout_execution_plan.json",
    "bytes": 25510,
    "sha256": "726ae764a1829db155bb5a03d01a247959cf491f6d333712df58e981c5a40f35",
    "payload_sha256": "02ac5205c7b59fbb2852ca9d2782263d9cf54526a4579656d74ad29239ebbbf0",
}
PREDECESSOR_DELIVERY = {
    "commit": "c668ba040cfcbc04905191169a6fc02b38b252bf",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 16362,
    "sha256": "208e425e09241e940dc3a071bd1ef41b45ef26d4faf37d462517486e2adb8e43",
    "payload_sha256": "cd210bf8ba6e5dca0c767510ab4fff7ff5c38cb0d895ffa4e71b7292324075a9",
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
        "status": "result-blind cache-identity schema correction after local preflight rejection",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "local_preflight": {
            "mode": "synthetic",
            "stage": "sealed identity validation before symlink staging",
            "failure_message": "synthetic-baseline-all-shards: cache identity schema mismatch",
            "provider_scorer_version_created": False,
        },
        "cause": (
            "the scoring stager required a reduced file/bytes/sha256 schema while the "
            "already-verified cache indexes intentionally retain immutable source and "
            "geometry identity fields"
        ),
        "correction": {
            "real_identity_fields": ["file", "bytes", "sha256", "source_image"],
            "synthetic_identity_fields": [
                "file",
                "bytes",
                "sha256",
                "name",
                "kind",
                "seed",
                "pitch_um",
                "papyrus",
                "kollesis",
            ],
            "matches_public_delivery_validator": True,
            "cache_job_index_manifest_or_npz_changed": False,
        },
        "scientific_gate": {
            "cache_npz_parsed_or_scientifically_inspected": False,
            "scientific_result_created_or_opened": False,
            "threshold_seed_panel_gate_or_claim_changed": False,
            "full_synthetic_staging_preflight_required_before_provider_retry": True,
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
        raise RuntimeError("execution plan is not the exact cache-plan predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("cache-plan predecessor payload mismatch")
    if "result_blind_scoring_job_plan_compatibility_correction" not in plan:
        raise RuntimeError("cache-job plan compatibility correction is absent")
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_scoring_cache_identity_schema_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_CACHE_IDENTITY_SCHEMA_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact cache-plan predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("cache-plan predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_scoring_cache_identity_schema_correction") != correction_record():
        raise RuntimeError("cache-identity schema correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scoring_cache_identity_schema_correction"] = {
        "public_execution_plan_commit": commit,
        "provider_scorer_version_created_for_failed_preflight": False,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery filename alias differs byte-for-byte")
    print("RESULT_BLIND_CACHE_IDENTITY_DELIVERY_CORRECTED")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
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
