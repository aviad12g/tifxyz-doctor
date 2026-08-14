#!/usr/bin/env python3
"""Freeze the result-blind CPU-provider projection allowlist correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "51101c314506f9b7a7cf3a5e3a7c985caae2d3f9",
    "file": "heldout_execution_plan.json",
    "bytes": 42198,
    "sha256": "3c529c058bda82746d3d325db0169f430637b53aa5f91e4efb1123a7c7ef9406",
    "payload_sha256": "82cf269002854401f59ad96882b9a98abbd77a69479b45d1d291e15d70102e3a",
}
PREDECESSOR_DELIVERY = {
    "commit": "3103a4d8b4825c52439cab5f965a5304c8ea33e4",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 19246,
    "sha256": "734d941916d0e5d3cc66a53dbc121651736d31136d0fec04382aee9f1e8b6ccb",
    "payload_sha256": "6b91f2e936119e61c756f118ccba136dd7da1d446b8ccf92fa7972a7d1c1e958",
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


def correction_record(public_tooling_commit: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind CPU-provider scientific-projection allowlist correction",
        "public_tooling_commit": require_commit(public_tooling_commit, "public tooling commit"),
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "failed_attempt": {
            "provider": "RunPod CPU",
            "pod_id": "h2xei44m0ee9m5",
            "stage": "plan/delivery validation before sealed cache staging",
            "failure_message": "corrected plan changes the cache-job scientific contract",
            "fixed_panel_renderer_started": False,
            "official_metric_scorer_started": False,
            "scientific_result_created_or_opened": False,
        },
        "cause": (
            "the scientific-projection allowlist did not classify the newly published "
            "result_blind_runpod_cpu_real_scoring record as operational metadata"
        ),
        "correction": {
            "cpu_provider_record_is_operational_metadata": True,
            "this_projection_record_is_operational_metadata": True,
            "scientific_projection_comparison_retained": True,
            "cache_job_plan_or_delivery_scientific_records_changed": False,
        },
        "scientific_gate": {
            "cache_npz_opened_or_inspected": False,
            "probability_or_endpoint_opened": False,
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "test_time_tuning_permitted": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("execution plan is not the exact CPU-projection predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("predecessor execution-plan payload mismatch")
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_runpod_cpu_projection_correction"] = correction_record(
        args.public_tooling_commit
    )
    write_hashed(args.plan, plan)
    print("RUNPOD_CPU_PROJECTION_PLAN_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        key: PREDECESSOR_DELIVERY[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("delivery is not the exact CPU-projection predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("predecessor cache-delivery payload mismatch")
    plan = load_hashed(args.plan)
    record = plan.get("result_blind_runpod_cpu_projection_correction")
    if record != correction_record(args.public_tooling_commit):
        raise RuntimeError("CPU projection correction is absent or changed")
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_runpod_cpu_projection_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": record["public_tooling_commit"],
        "retry_permitted_after_full_preflight": True,
        "scientific_contract_changed": False,
        "scientific_result_created_or_opened": False,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("RUNPOD_CPU_PROJECTION_DELIVERY_CORRECTED")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
    plan.add_argument("--public-tooling-commit", required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = subparsers.add_parser("delivery")
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
