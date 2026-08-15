#!/usr/bin/env python3
"""Freeze the exact public metric-worker sibling after a result-blind failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR = {
    "commit": "c90b2277d98edea347189d7b9fba45baee065620",
    "file": "heldout_execution_plan.json",
    "bytes": 60217,
    "sha256": "817bffb0cdd5528e89d200744219e08ee35d09e5bba57419149861da53cfd54f",
    "payload_sha256": "34c91a427f7543624d28bad6707a922be4214391906016a5e3cfe512a2596302",
}
DELIVERY_PREDECESSOR = {
    "commit": "f78479ed04f0958bb1b13876ad51809f9d08c13c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 23891,
    "sha256": "726c887561e15fc70959933d42f9cf603afc31956bbe5820068c771ec47e6e4e",
    "payload_sha256": "6f266419ef2494d75f8cfdcba60c3ccd3e957065c3fd90b50d2a4fe277a401a3",
}
FAILED_RECEIPT = {
    "file": "provider_receipt.json",
    "payload_sha256": "9f1a1a4371e4145b0bb5e09560aa880ac153b18c27ce63219d5e7513afa6b028",
    "pod_id": "g6napb4zi036b1",
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


def correction(args: argparse.Namespace) -> dict:
    return {
        "schema_version": "1.0",
        "status": "exact public official-metric sibling frozen before retry",
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "corrected_launcher": identity(args.launcher),
        "corrected_stager": identity(args.stager),
        "public_metric_worker": identity(args.source_root / "official_metric.py"),
        "failed_attempt": FAILED_RECEIPT | {
            "stage": "first independent official-metric subprocess after verified private transport",
            "failure_type": "FileNotFoundError",
            "failure_message": "official_metric.py sibling worker absent",
            "private_transport_verified": True,
            "scientific_outputs_inspected": False,
        },
        "scientific_contract": {
            "metric_worker_source_bytes_changed": False,
            "scorer_default_worker_path_changed": False,
            "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
            "npz_panel_probability_endpoint_or_result_opened": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PLAN_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong metric-worker predecessor plan")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor plan payload mismatch")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_scorer_public_metric_worker_correction"] = correction(args)
    write_hashed(args.plan, plan)
    print("SCORER_PUBLIC_METRIC_WORKER_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {key: DELIVERY_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong metric-worker predecessor delivery")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scorer_public_metric_worker_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "exact_public_metric_worker_materialized_beside_scorer": True,
        "scientific_contract_changed": False,
        "scientific_output_opened_or_used": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("SCORER_PUBLIC_METRIC_WORKER_DELIVERY_FROZEN")


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
