#!/usr/bin/env python3
"""Freeze the proven-equivalent concurrent real scorer before held-out retry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREDECESSOR_PLAN = {
    "file": "heldout_execution_plan.json",
    "bytes": 52907,
    "sha256": "bed684b2fb1d4d505bffc8bc4350036d7745dd90acb4031f18018f753f4cc560",
    "payload_sha256": "0154df3bcf8568b5c52ad36e9d9eb68594062d57f78c5426bf7c9015129be096",
}
PREDECESSOR_DELIVERY = {
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 21422,
    "sha256": "6033e8a5e4309d1b4b0f8223f63f0c3215e1866269e00517f9bb83ab315717b1",
    "payload_sha256": "f07325081390f8c88cfd9531bbbeb28fc053b6e27b954af21d9fb1d2e3117210",
}
EXACT_SCORER = {
    "file": "score_real_test.py",
    "sha256": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
}
EQUIVALENCE_COMMIT = "85b0190613ea35b73f16079a67ae52d9eaa9cf8e"
PARALLEL_WORKERS = 32


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


def script_identity(path: Path) -> dict:
    return {"file": path.name, "sha256": sha256_file(path)}


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("public commit must be 40 lowercase hexadecimal characters")
    return value


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_hashed(path)


def equivalence_identity(path: Path) -> dict:
    report = load_hashed(path)
    if report.get("status") != "public official-metric parallel scorer equivalence verified":
        raise RuntimeError("parallel equivalence report status mismatch")
    if any(
        not comparison.get("exact_python_value_equality")
        or not comparison.get("frozen_cache_order_preserved")
        or comparison.get("result_sha256") != report.get("sequential_result_sha256")
        for comparison in report.get("comparisons", [])
    ):
        raise RuntimeError("parallel equivalence report contains a mismatch")
    if [record.get("parallel_workers") for record in report["comparisons"]] != [4, 16, 32]:
        raise RuntimeError("parallel equivalence worker matrix mismatch")
    return identity(path) | {
        "commit": EQUIVALENCE_COMMIT,
        "payload_sha256": report["payload_sha256"],
    }


def correction(report: Path) -> dict:
    return {
        "schema_version": "1.0",
        "status": "result-blind verified parallel real scoring frozen before retry",
        "parallel_workers": PARALLEL_WORKERS,
        "equivalence_report": equivalence_identity(report),
        "exact_predecessor_scorer": EXACT_SCORER,
        "scientific_contract": {
            "per_cache_metric_subprocess_changed": False,
            "cache_threshold_or_input_changed": False,
            "aggregation_bootstrap_or_serialization_changed": False,
            "frozen_cache_order_preserved": True,
            "test_time_tuning_permitted": False,
            "only_independent_subprocess_scheduling_changed": True,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {
        key: PREDECESSOR_PLAN[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("execution plan is not the exact predecessor")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PREDECESSOR_PLAN["payload_sha256"]:
        raise RuntimeError("execution-plan predecessor payload mismatch")
    if plan["one_shot_scorers"]["real"]["script"] != EXACT_SCORER:
        raise RuntimeError("execution plan does not use the exact predecessor scorer")
    plan["one_shot_scoring_launcher"] = identity(args.launcher)
    plan["one_shot_scorers"]["real"]["script"] = script_identity(args.parallel_scorer)
    plan["result_blind_verified_parallel_real_scoring"] = correction(args.equivalence_report)
    write_hashed(args.plan, plan)
    print("VERIFIED_PARALLEL_REAL_SCORING_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {
        key: PREDECESSOR_DELIVERY[key] for key in ("file", "bytes", "sha256")
    }:
        raise RuntimeError("delivery is not the exact predecessor")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != PREDECESSOR_DELIVERY["payload_sha256"]:
        raise RuntimeError("delivery predecessor payload mismatch")
    plan = load_hashed(args.plan)
    expected_correction = correction(args.equivalence_report)
    if plan.get("result_blind_verified_parallel_real_scoring") != expected_correction:
        raise RuntimeError("verified parallel correction is absent or changed")
    commit = require_commit(args.public_plan_commit)
    delivery["public_execution_plan"] = {
        "commit": commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": sha256_file(args.plan),
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_verified_parallel_real_scoring"] = {
        "public_execution_plan_commit": commit,
        "equivalence_report": expected_correction["equivalence_report"],
        "parallel_workers": PARALLEL_WORKERS,
        "scientific_contract_changed": False,
        "held_out_result_opened_or_used": False,
        "real_retry_permitted": True,
    }
    write_hashed(args.delivery, delivery)
    print("VERIFIED_PARALLEL_REAL_SCORING_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--launcher", type=Path, required=True)
    plan.add_argument("--parallel-scorer", type=Path, required=True)
    plan.add_argument("--equivalence-report", type=Path, required=True)
    plan.set_defaults(function=freeze_plan)
    delivery = commands.add_parser("delivery")
    delivery.add_argument("--delivery", type=Path, required=True)
    delivery.add_argument("--plan", type=Path, required=True)
    delivery.add_argument("--public-plan-commit", required=True)
    delivery.add_argument("--equivalence-report", type=Path, required=True)
    delivery.set_defaults(function=freeze_delivery)
    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
