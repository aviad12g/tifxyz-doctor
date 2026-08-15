#!/usr/bin/env python3
"""Freeze the verified-parallel scorer scientific-projection correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLAN_PREDECESSOR = {
    "commit": "6f0a49bb99592098f980ce57fdbc6d62578af352",
    "file": "heldout_execution_plan.json",
    "bytes": 55068,
    "sha256": "b1d00dbf940db8397044050e19044563af26449dac3dc52062d8fb7727f5e57c",
    "payload_sha256": "e039587d21b9b7265cb36ea3101e648904df1f8728c6c86e84190b518800429f",
}
DELIVERY_PREDECESSOR = {
    "commit": "c05f5e58e643be23dc5dd7479f33774f85a62cc0",
    "file": "heldout_cache_delivery_manifest.json",
    "bytes": 22715,
    "sha256": "51c7062875c14a5a810037205acf1184511289be7e72bf7d9d9ecce742334fbc",
    "payload_sha256": "f4ef8da4a2ac423949772894f1c03280bd8c40c44b6783feb1c989bf81340109",
}
FAILED_RECEIPT = {
    "file": "provider_receipt.json",
    "payload_sha256": "190ee66c4c9e7172fdf608ce058e138028ba07813ae36def747e699e59303365",
    "pod_id": "1f8axs6hidbwrm",
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


def correction(stager: Path, tooling_commit: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "verified-parallel scorer identity projected to exact predecessor for cache-job comparison",
        "public_tooling_commit": require_commit(tooling_commit, "public tooling commit"),
        "corrected_scoring_stager": identity(stager),
        "failed_attempt": FAILED_RECEIPT | {
            "stage": "held-out staging before panels or scoring",
            "failure_message": "corrected plan changes the cache-job scientific contract",
            "scientific_outputs_inspected": False,
        },
        "projection": {
            "current_verified_parallel_scorer": {
                "file": "score_real_test_parallel.py",
                "sha256": "f280c5ed7c74df27d9986757e05109d358caf429e23e4698fb44c91616387f7c",
            },
            "exact_predecessor_scorer": {
                "file": "score_real_test.py",
                "sha256": "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79",
            },
            "projection_scope": "one_shot_scorers.real.script",
        },
        "scientific_contract": {
            "only_independent_subprocess_scheduling_changed": True,
            "per_cache_metric_subprocess_changed": False,
            "aggregation_bootstrap_serialization_or_cache_order_changed": False,
            "held_out_result_opened_or_used": False,
        },
    }


def freeze_plan(args: argparse.Namespace) -> None:
    if identity(args.plan) != {key: PLAN_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong verified-parallel projection predecessor plan")
    plan = load_hashed(args.plan)
    if plan["payload_sha256"] != PLAN_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor plan payload mismatch")
    plan["one_shot_scoring_stager"] = identity(args.stager)
    plan["result_blind_verified_parallel_scorer_projection_correction"] = correction(
        args.stager, args.public_tooling_commit
    )
    write_hashed(args.plan, plan)
    print("VERIFIED_PARALLEL_SCORER_PROJECTION_PLAN_FROZEN")


def freeze_delivery(args: argparse.Namespace) -> None:
    if identity(args.delivery) != {key: DELIVERY_PREDECESSOR[key] for key in ("file", "bytes", "sha256")}:
        raise RuntimeError("wrong verified-parallel projection predecessor delivery")
    delivery = load_hashed(args.delivery)
    if delivery["payload_sha256"] != DELIVERY_PREDECESSOR["payload_sha256"]:
        raise RuntimeError("predecessor delivery payload mismatch")
    plan = load_hashed(args.plan)
    public_plan_commit = require_commit(args.public_plan_commit, "public plan commit")
    delivery["public_execution_plan"] = identity(args.plan) | {
        "commit": public_plan_commit,
        "payload_sha256": plan["payload_sha256"],
    }
    delivery["result_blind_verified_parallel_scorer_projection_correction"] = {
        "public_execution_plan_commit": public_plan_commit,
        "public_tooling_commit": require_commit(args.public_tooling_commit, "public tooling commit"),
        "scientific_contract_changed": False,
        "held_out_result_opened_or_used": False,
        "retry_permitted_after_full_preflight": True,
    }
    write_hashed(args.delivery, delivery)
    args.alias.write_bytes(args.delivery.read_bytes())
    if sha256_file(args.alias) != sha256_file(args.delivery):
        raise RuntimeError("delivery alias differs byte-for-byte")
    print("VERIFIED_PARALLEL_SCORER_PROJECTION_DELIVERY_FROZEN")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--stager", type=Path, required=True)
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
