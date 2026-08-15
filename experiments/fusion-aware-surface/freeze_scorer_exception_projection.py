#!/usr/bin/env python3
"""Freeze bounded result-blind scorer-exception projection in the launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR = {
    "commit": "03353db5431c685d038b43ad3bef44e53ed883aa",
    "file": "heldout_execution_plan.json",
    "bytes": 56633,
    "sha256": "3d9eafa2fc56dc899557d6a732cb2c83d33e22e9c3d9a99a3cc723686e0451dc",
    "payload_sha256": "1231137bbd068be19a83c209b3a2d269b7bbcb96cbc08496b642d181e4ea309d",
}
DELIVERY_PREDECESSOR = {
    "commit": "a0da4a8d470be7f189842304003bb80f45bb8dc3",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 23076,
    "sha256": "c9ea26691b507713e0b7d23352d3e39602b2d29a6ff94f073ad26de23b7d0131",
    "payload_sha256": "7034898c7b5c612904db495f9e64c5acd82efa055f8a6e15f4197acfa4b41561",
}
FAILED_RECEIPT = {
    "file": "provider_receipt.json",
    "payload_sha256": "ff6d5dfc169e14b8159a0bf47ba9aed1c44d47c3aa5d9a16a6d7386236eef7a0",
    "pod_id": "k0fxlidhmzyi23",
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
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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


def correction(launcher: Path, tooling_commit: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "bounded scorer stderr exception projection frozen before retry",
        "public_tooling_commit": require_commit(tooling_commit, "public tooling commit"),
        "corrected_launcher": identity(launcher),
        "failed_attempt": FAILED_RECEIPT | {
            "stage": "exact real scorer subprocess after complete staging",
            "failure_message": "one-shot scorer failed; scientific stdout remains sealed",
            "scientific_outputs_inspected": False,
        },
        "projection": {
            "stream": "stderr",
            "maximum_exception_lines": 3,
            "maximum_line_bytes": 1_024,
            "scientific_stdout_projected": False,
        },
        "scientific_contract": {
            "scorer_source_changed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "held_out_result_opened_or_used": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PLAN_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong scorer-exception predecessor plan")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["result_blind_scorer_exception_projection_correction"] = correction(
        args.launcher, args.public_tooling_commit
    )
    write_hashed(args.plan, plan)
    print("SCORER_EXCEPTION_PROJECTION_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {key: DELIVERY_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong scorer-exception predecessor delivery")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scorer_exception_projection_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "scientific_stdout_projected": False,
        "scientific_contract_changed": False,
        "held_out_result_opened_or_used": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("SCORER_EXCEPTION_PROJECTION_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--public-tooling-commit", required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = commands.add_parser("delivery")
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
