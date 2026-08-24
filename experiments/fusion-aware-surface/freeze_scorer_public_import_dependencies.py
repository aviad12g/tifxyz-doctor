#!/usr/bin/env python3
"""Freeze exact public scorer-import dependencies after a result-blind failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR = {
    "commit": "0aaa9d53976b330054815ae945010e9fe13e80e9",
    "file": "heldout_execution_plan.json",
    "bytes": 58034,
    "sha256": "930b995258e335ff6a3afe9a999074e370c5e20adb0e6a5874882dc3299592ce",
    "payload_sha256": "f5689728be502080b678dc91f677fe8d3465154f6245fe5c04cd7d6e8a5643d9",
}
DELIVERY_PREDECESSOR = {
    "commit": "c84b6da55824f89563622dc9e50a086a05a1994b",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 23471,
    "sha256": "666b34b2349b98e4da2075894f434f891e8a438ff6b4b157f8aa81647ffa39bd",
    "payload_sha256": "adbcd8100c3834706f1fafbb461e7d11084dd07cb9f27764fb78747fa989e171",
}
FAILED_RECEIPT = {
    "file": "provider_receipt.json",
    "payload_sha256": "9c347768feeef2123b49f33f5db6e86920b5fbcfc2435d18465b6ca262c8e2b6",
    "pod_id": "djai7jtnk3helh",
}
DEPENDENCIES = (
    "fusion_loss.py",
    "gap_supervision.py",
    "inference.py",
    "normalization.py",
    "train_fusion_aware.py",
)


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


def correction(args: argparse.Namespace) -> dict:
    return {
        "schema_version": "1.0",
        "status": "exact public scorer import dependencies frozen before retry",
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "corrected_launcher": identity(args.launcher),
        "corrected_stager": identity(args.stager),
        "public_import_dependencies": [identity(args.source_root / name) for name in DEPENDENCIES],
        "failed_attempt": FAILED_RECEIPT | {
            "stage": "exact real scorer after verified private signed-bundle transport",
            "failure_type": "ModuleNotFoundError",
            "failure_message": "No module named 'inference'",
            "private_transport_verified": True,
            "scientific_outputs_inspected": False,
        },
        "scientific_contract": {
            "dependency_source_bytes_changed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PLAN_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong public-import predecessor plan")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_scorer_public_import_dependency_correction"] = correction(args)
    write_hashed(args.plan, plan)
    print("SCORER_PUBLIC_IMPORT_DEPENDENCY_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {key: DELIVERY_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong public-import predecessor delivery")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scorer_public_import_dependency_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "exact_public_dependencies_fetched_before_scorer": True,
        "scientific_contract_changed": False,
        "scientific_output_opened_or_used": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("SCORER_PUBLIC_IMPORT_DEPENDENCY_DELIVERY_FROZEN")


def add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--launcher", type=Path, required=True)
    command.add_argument("--stager", type=Path, required=True)
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--public-tooling-commit", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    add_common(plan)
    plan.set_defaults(function=freeze_plan)
    delivery = commands.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--alias", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    add_common(delivery)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
