#!/usr/bin/env python3
"""Freeze the operational projection for verified parallel scorer metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR_PAYLOAD = "147a766f8a7a9690dc055960c12eb7d0b111c25b55ec45c87165b0691937a274"
DELIVERY_PREDECESSOR_PAYLOAD = "fcbbd34d59404600197071287a23dbc2556c76bd5cbb5857b396e8bf6b87b74c"


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


def correction(stager: Path) -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind verified parallel records classified as operational metadata",
        "corrected_scoring_stager": identity(stager),
        "operational_fields_added": [
            "result_blind_verified_parallel_real_scoring",
            "result_blind_public_commit_binding_correction",
            "result_blind_verified_parallel_projection_correction",
        ],
        "scientific_contract_changed": False,
        "held_out_result_opened_or_used": False,
    }


def freeze_plan(args: argparse.Namespace) -> None:
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong execution-plan projection predecessor")
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_verified_parallel_projection_correction"] = correction(args.stager)
    write_hashed(args.plan, plan)
    print("VERIFIED_PARALLEL_PROJECTION_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong delivery projection predecessor")
    plan = load_hashed(args.plan)
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_verified_parallel_projection_correction"] = {
        "public_execution_plan_commit": commit,
        "scientific_contract_changed": False,
        "held_out_result_opened_or_used": False,
    }
    write_hashed(args.delivery, delivery)
    print("VERIFIED_PARALLEL_PROJECTION_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = commands.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
