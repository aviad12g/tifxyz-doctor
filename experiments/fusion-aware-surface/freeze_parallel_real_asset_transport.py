#!/usr/bin/env python3
"""Freeze result-blind public transport for the accelerated real scorer source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "6540d1044cecf3a9a686daf207b2362c37be66a3",
    "file": "heldout_execution_plan.json",
    "bytes": 34736,
    "sha256": "f425a1c33bd3597b81b5734912bcbf5ef7f9a1fcd5382aa12a9c2802c543d548",
    "payload_sha256": "738383be47566aadc9b9695972c1ae6dae71e59fd443aa515d88a57999fb629c",
}
PREDECESSOR_DELIVERY = {
    "commit": "012b0bb0587a217f737edb4176469bfde776ae60",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 18296,
    "sha256": "d2a725e5f71afc0d01813e4275927f4c153b6cd3c7420a4fd029e21c5c9af9fe",
    "payload_sha256": "5de4ab189a35bbc882553cfa52354e6a77d61c72d77efa246b43bcd78ed1e953",
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


def correction_record() -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind accelerated real scorer public-source transport correction",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "failed_attempt": {
            "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
            "kernel_version": 5,
            "status": "ERROR",
            "elapsed_seconds": 6,
            "failure_stage": "asset-ledger validation before held-out staging",
            "message": "project source is not asset-ledger bound: score_real_test.py",
        },
        "correction": {
            "real_scorer_fetched_from_public_plan_commit": True,
            "real_scorer_sha256_verified_before_execution": True,
            "immutable_training_asset_ledger_changed": False,
            "synthetic_public_scorer_transport_pattern_reused": True,
        },
        "scientific_gate": {
            "held_out_input_staging_started_in_failed_attempt": False,
            "panel_renderer_started_in_failed_attempt": False,
            "scorer_invocation_started_in_failed_attempt": False,
            "scientific_result_created_or_opened": False,
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
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
        raise RuntimeError("plan is not the exact parallel-retry predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("parallel-retry predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["result_blind_parallel_real_asset_transport"] = correction_record()
    write_hashed(args.plan, plan)
    print("PARALLEL_REAL_PUBLIC_SOURCE_TRANSPORT_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact parallel-retry predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("parallel-retry predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_parallel_real_asset_transport") != correction_record():
        raise RuntimeError("real public-source transport correction is absent")
    commit = args.public_plan_commit
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal characters")
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_parallel_real_asset_transport"] = {
        "public_execution_plan_commit": commit,
        "failed_real_version": 5,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "real_v6_retry_permitted": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("PARALLEL_REAL_PUBLIC_SOURCE_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
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
