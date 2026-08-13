#!/usr/bin/env python3
"""Freeze the result-blind Python 3.12 scorer runtime transport correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "commit": "e36169e53acbf83ae77642dc47b00c6f98a24ec1",
    "file": "heldout_execution_plan.json",
    "bytes": 27580,
    "sha256": "176291c91b0161e5761dc6efddfca1e2513166fd2893d3fea768b58f3cddd04a",
    "payload_sha256": "d8cc6199483121145edddf141d33384b323e6a320448209394a0061265f9d361",
}
PREDECESSOR_DELIVERY = {
    "commit": "7fe1c06671b56c66554ca377d4808af3cf76187c",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 16716,
    "sha256": "bab108e680dd493357361690b206c50d76ed63c1cfa99b4fd50e528c676ab47c",
    "payload_sha256": "607c83abc2db98f030b4b9802ccc467764b99ea525341b876a285c99cb502f08",
}
FAILED_ATTEMPTS = [
    {
        "mode": "real",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-real-scoring",
        "kernel_version": 3,
    },
    {
        "mode": "synthetic",
        "kernel_id": "aviadcohen1/vesuvius-fusion-one-shot-synthetic-scoring",
        "kernel_version": 3,
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
        "status": "result-blind Python 3.12 scorer runtime transport correction",
        "predecessor_execution_plan": PREDECESSOR_PLAN,
        "predecessor_cache_delivery": PREDECESSOR_DELIVERY,
        "failed_attempts": FAILED_ATTEMPTS,
        "failure_stage": "runtime installation before held-out input staging",
        "failure_message": (
            "no matching distribution for the cu121 torch dependency "
            "nvidia-cudnn-cu12==9.1.0.70"
        ),
        "cause": (
            "the frozen CUDA 12.1 PyTorch wheel requested a transitive cuDNN build "
            "no longer present on the provider index; the scorer kernels request no GPU "
            "and consume already-produced caches"
        ),
        "correction": {
            "torch": "2.5.1+cpu",
            "torchvision": "0.20.1+cpu",
            "other_runtime_versions_changed": False,
            "primary_index": "https://download.pytorch.org/whl/cpu",
            "extra_index": "https://pypi.org/simple",
            "binary_wheels_only": True,
            "scorer_kernel_accelerator": "none",
            "cache_manifest_or_npz_changed": False,
        },
        "python312_linux_resolution_preflight": {
            "status": "all exact top-level requirements and transitive wheels resolved",
            "files": 30,
            "bytes": 260266954,
            "sorted_wheel_ledger_sha256": (
                "7e8967a004c7e6e97a1e223f33f8edb78731c1d1dfa7c21ea52df9c63747bfec"
            ),
            "platform_tags": ["linux_x86_64", "manylinux2014_x86_64"],
            "python_version": "3.12",
            "implementation": "cp",
            "abi": "cp312",
        },
        "scientific_gate": {
            "held_out_input_staging_started_in_failed_attempts": False,
            "scorer_invocation_started_in_failed_attempts": False,
            "cache_npz_opened_or_inspected": False,
            "scientific_result_created_or_opened": False,
            "threshold_model_seed_panel_endpoint_gate_or_claim_changed": False,
            "runtime_package_versions_changed": False,
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
        raise RuntimeError("execution plan is not the exact runtime predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("runtime predecessor plan payload mismatch")
    if "result_blind_scoring_cache_identity_schema_correction" not in plan:
        raise RuntimeError("cache-identity correction is absent")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["result_blind_scoring_runtime_transport_correction"] = correction_record()
    write_hashed(args.plan, plan)
    print("RESULT_BLIND_SCORING_RUNTIME_TRANSPORT_CORRECTED")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        "file": args.delivery.name,
        "bytes": PREDECESSOR_DELIVERY["bytes"],
        "sha256": PREDECESSOR_DELIVERY["sha256"],
    }:
        raise RuntimeError("delivery is not the exact runtime predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("runtime predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    if plan.get("result_blind_scoring_runtime_transport_correction") != correction_record():
        raise RuntimeError("runtime transport correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_scoring_runtime_transport_correction"] = {
        "public_execution_plan_commit": commit,
        "failed_attempts": FAILED_ATTEMPTS,
        "scientific_result_created_or_opened": False,
        "scientific_contract_changed": False,
        "retry_permitted_after_exact_wheel_resolution": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery filename alias differs byte-for-byte")
    print("RESULT_BLIND_SCORING_RUNTIME_DELIVERY_CORRECTED")


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
