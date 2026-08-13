#!/usr/bin/env python3
"""Freeze the result-blind runtime-record scientific projection correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "251ffa9111cc7f5d5977ddc26006c68fc1b0c0af",
    "file": "heldout_execution_plan.json",
    "bytes": 30362,
    "sha256": "ef1be4c5bba0f165602d6885fabdd0e308d79059034863965e9d561e6a4266a3",
    "payload_sha256": "a9d317bd50df2adb0d1e2b6e25103d0d8e9ce3b75efce170f81c0818f8b5880f",
}
PREDECESSOR_DELIVERY = {
    "commit": "76f60591b72b4ec0b3def6d4bc2cc551722b0e3c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 17334,
    "sha256": "2b8f1c659acf283f0c45ab2b868479bd27ed810914b00774013d2975b281cb10",
    "payload_sha256": "f44893fb8d0107970f2e9e75e5226dab65b1ca4cd1d115ffcc2abaa31fe52efa",
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
        "status": "result-blind runtime-record scientific projection correction",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "local_preflight": {
            "stage": "plan/delivery validation before sealed cache staging",
            "failure_message": "corrected plan changes the cache-job scientific contract",
            "provider_scorer_version_created": False,
        },
        "cause": (
            "the scientific projection allowlist predated the published runtime transport "
            "record and therefore compared that operational record to the original cache plan"
        ),
        "correction": {
            "runtime_transport_record_is_operational_metadata": True,
            "runtime_projection_record_is_operational_metadata": True,
            "scientific_projection_comparison_retained": True,
            "cache_job_plan_or_delivery_records_changed": False,
        },
        "scientific_gate": {
            "cache_staging_started_in_failed_preflight": False,
            "cache_npz_opened_or_inspected": False,
            "scorer_invocation_started": False,
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
        raise RuntimeError("execution plan is not the exact projection predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("projection predecessor plan payload mismatch")
    if "result_blind_scoring_runtime_transport_correction" not in plan:
        raise RuntimeError("runtime transport correction is absent")
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_scoring_runtime_projection_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_SCORING_RUNTIME_PROJECTION_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact projection predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("projection predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_scoring_runtime_projection_correction") != correction_record():
        raise RuntimeError("runtime projection correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scoring_runtime_projection_correction"] = {
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
    print("RESULT_BLIND_SCORING_RUNTIME_PROJECTION_DELIVERY_CORRECTED")


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
