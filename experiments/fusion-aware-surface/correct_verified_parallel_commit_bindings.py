#!/usr/bin/env python3
"""Correct locally predicted Git commit bindings before any provider retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR_PAYLOAD = "673dd7bfe414a43df28d1beebe82c9e0d5aed4da5d45b1eedd69878309e05fcf"
DELIVERY_PREDECESSOR_PAYLOAD = "22a9caec9f87b059e86f3812485e1bad008d2f3cef1a3da4705d10d881f72f86"
ACTUAL_EQUIVALENCE_COMMIT = "85b0190613ea35b73f16079a67ae52d9eaa9cf8e"
ERRONEOUS_EQUIVALENCE_COMMIT = "85b0190541e402c1c20e6f7870da5669ed756521"


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


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal characters")
    return value


def correction_record() -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind local Git commit binding corrected before provider retry",
        "erroneous_unpublished_equivalence_commit": ERRONEOUS_EQUIVALENCE_COMMIT,
        "actual_equivalence_commit": ACTUAL_EQUIVALENCE_COMMIT,
        "scientific_outputs_opened_or_used": False,
        "scientific_contract_changed": False,
    }


def freeze_plan(args: argparse.Namespace) -> None:
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong execution-plan predecessor payload")
    equivalence = plan["result_blind_verified_parallel_real_scoring"]["equivalence_report"]
    if equivalence.get("commit") != ERRONEOUS_EQUIVALENCE_COMMIT:
        raise RuntimeError("predicted equivalence commit is not present")
    equivalence["commit"] = ACTUAL_EQUIVALENCE_COMMIT
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["result_blind_public_commit_binding_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("VERIFIED_PARALLEL_COMMIT_BINDING_PLAN_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR_PAYLOAD:
        raise RuntimeError("wrong delivery predecessor payload")
    plan = load_hashed(args.plan)
    equivalence = delivery["result_blind_verified_parallel_real_scoring"][
        "equivalence_report"
    ]
    if equivalence.get("commit") != ERRONEOUS_EQUIVALENCE_COMMIT:
        raise RuntimeError("predicted delivery equivalence commit is not present")
    equivalence["commit"] = ACTUAL_EQUIVALENCE_COMMIT
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_verified_parallel_real_scoring"][
        "public_execution_plan_commit"
    ] = commit
    delivery["result_blind_public_commit_binding_correction"] = correction_record()
    write_hashed(args.delivery, delivery)
    print("VERIFIED_PARALLEL_COMMIT_BINDING_DELIVERY_CORRECTED")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
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
