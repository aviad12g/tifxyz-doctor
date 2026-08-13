#!/usr/bin/env python3
"""Publish a result-blind correction for the one-shot scoring handoff.

The first scorer pair failed before input staging because the corrected public
synthetic scorer was not present in the older immutable training-asset ledger.
This helper changes only the scorer transport/provenance binding: the corrected
scorer is fetched from the public plan commit and hash checked. It does not
change caches, thresholds, seeds, panels, gates, or result interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "46d030245c14ecb2be63796a308581c462fc2e29",
    "file": "heldout_execution_plan.json",
    "bytes": 20443,
    "sha256": "29d08a7430c8870582a96579fcc77737ac699aef64efc159ba94ede87009da5d",
    "payload_sha256": "3826bf641941b5d70fa52f9a23fc1711ae538be8fc974dbba1b5f0db8f18afdd",
}
PREDECESSOR_DELIVERY = {
    "commit": "7eb2fd90e33f0bf3b3fad91fcce8e7b98f8b8413",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 14848,
    "sha256": "80929fb81c4f2a13a3e6e28fac3eaff2253cd7ea6243767a1b7d5aa47324870f",
    "payload_sha256": "cf2292c6a596338116c63e3be13c3fe5d03c9df5f6017a7b7feac1c2f3ca50cc",
}
CORRECTED_SCORER = {
    "file": "score_synthetic_test_v2.py",
    "bytes": 10336,
    "sha256": "d594cea7d58b08bbeccab5ec65f0a3d64191a70d07e9423314cd607d9fe53d05",
    "first_public_commit": "1e0ced2c928fa0842aa8c601f39dbcd49272247e",
}
FAILED_ATTEMPTS = [
    {
        "mode": "real",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
        "kernel_version": 1,
    },
    {
        "mode": "synthetic",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
        "kernel_version": 1,
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
        "status": "result-blind scorer transport correction after fail-closed pre-staging rejection",
        "predecessor_public_execution_plan": PREDECESSOR_PLAN,
        "predecessor_public_cache_delivery": PREDECESSOR_DELIVERY,
        "failed_attempts": FAILED_ATTEMPTS,
        "failure_stage": "asset-ledger validation before held-out input staging",
        "failure_message": (
            "project source is not asset-ledger bound: score_synthetic_test_v2.py"
        ),
        "cause": {
            "asset_dataset_source": "aviadcohen1/vesuvius-fusion-aware-training-assets/1",
            "asset_ledger_records": 278,
            "asset_ledger_sha256": (
                "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba"
            ),
            "corrected_scorer_absent_from_immutable_asset_ledger": True,
        },
        "correction": {
            "corrected_scorer": CORRECTED_SCORER,
            "source": "hash-bound fetch from the public execution-plan commit",
            "asset_dataset_version_changed": False,
            "canonical_provider_kernel_ids_used": True,
        },
        "scientific_gate": {
            "held_out_input_staging_started_in_failed_attempts": False,
            "scorer_invocation_started_in_failed_attempts": False,
            "cache_npz_opened_or_inspected": False,
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
        raise RuntimeError("execution plan is not the exact predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("execution-plan payload differs from the predecessor")
    if plan.get("one_shot_scorers", {}).get("synthetic", {}).get("script") != {
        "file": CORRECTED_SCORER["file"],
        "sha256": CORRECTED_SCORER["sha256"],
    }:
        raise RuntimeError("synthetic scorer contract differs from the frozen correction")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["scoring_package_generator"] = identity(args.generator)
    plan["scoring_pair_controller"] = identity(args.controller)
    plan["final_result_validator"] = identity(args.validator)
    plan["result_blind_scoring_asset_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_SCORING_ASSET_PLAN_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("cache delivery is not the exact predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("cache-delivery payload differs from the predecessor")
    plan = load_hashed(args.plan)
    correction = plan.get("result_blind_scoring_asset_correction")
    if correction != correction_record():
        raise RuntimeError("corrected plan provenance is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scoring_asset_correction"] = {
        "public_execution_plan_commit": commit,
        "failed_attempts": FAILED_ATTEMPTS,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "retry_permitted": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if identity(args.alias) != {
        "file": args.alias.name,
        "bytes": args.delivery.stat().st_size,
        "sha256": sha256_file(args.delivery),
    }:
        raise RuntimeError("delivery filename alias differs byte-for-byte")
    print("RESULT_BLIND_SCORING_ASSET_DELIVERY_CORRECTED")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--generator", type=Path, required=True)
    plan.add_argument("--controller", type=Path, required=True)
    plan.add_argument("--validator", type=Path, required=True)
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
