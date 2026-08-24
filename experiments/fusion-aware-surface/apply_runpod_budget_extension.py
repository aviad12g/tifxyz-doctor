#!/usr/bin/env python3
"""Apply a public result-blind RunPod budget extension to a provider receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(payload: dict) -> str:
    body = dict(payload)
    body.pop("payload_sha256", None)
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = canonical_sha256(payload)
    if payload.get("payload_sha256") != observed:
        raise RuntimeError(f"budget-extension payload mismatch: {observed}")
    return payload


def require_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("public extension commit must be a full lowercase SHA-1")
    return value


def validate(extension: dict, receipt: dict) -> bool:
    if extension.get("status") != (
        "result-blind operational budget extension authorized before primary completion"
    ):
        raise RuntimeError("unexpected budget-extension status")
    frozen = extension["frozen_primary"]
    if receipt.get("plan_payload_sha256") != frozen["original_plan_payload_sha256"]:
        raise RuntimeError("receipt is bound to another scientific plan")
    if receipt.get("public_replacement_commit") != frozen["original_public_replacement_commit"]:
        raise RuntimeError("receipt is bound to another public replacement commit")
    pods = receipt.get("pods")
    if not isinstance(pods, list) or [pod.get("id") for pod in pods] != [frozen["pod_id"]]:
        raise RuntimeError("receipt is bound to another RunPod allocation")
    jobs = pods[0].get("jobs")
    if not isinstance(jobs, list) or len(jobs) != frozen["job_count"]:
        raise RuntimeError("receipt job count differs from frozen primary")
    if float(receipt.get("total_hourly_rate_usd")) != frozen["active_aggregate_hourly_rate_usd"]:
        raise RuntimeError("receipt hourly rate differs from authorized allocation")
    if receipt.get("scientific_outputs_inspected") is not False:
        raise RuntimeError("budget extension must be applied before scientific output access")
    if receipt.get("status") != "PRIMARY_EXECUTOR_RUNNING_SEALED":
        raise RuntimeError("budget extension requires a sealed running primary")
    budget = extension["budget"]
    existing = receipt.get("budget_extensions", [])
    if not isinstance(existing, list):
        raise RuntimeError("receipt budget-extension history is not a list")
    predecessor = extension.get("predecessor_budget_extension")
    expected_prior_count = 0
    if predecessor is not None:
        expected_prior_count = predecessor["expected_prior_extension_count"]
        if expected_prior_count < 1:
            raise RuntimeError("invalid predecessor extension count")
    already_applied = (
        len(existing) == expected_prior_count + 1
        and existing[-1].get("extension_payload_sha256") == extension["payload_sha256"]
    )
    if already_applied:
        if float(receipt.get("billing_cutoff_usd")) != budget["new_billing_cutoff_usd"]:
            raise RuntimeError("applied receipt billing cutoff differs from extension")
        if float(receipt.get("hard_total_cap_usd")) != budget["new_hard_total_cap_usd"]:
            raise RuntimeError("applied receipt hard cap differs from extension")
        return True
    if len(existing) != expected_prior_count:
        raise RuntimeError("receipt budget-extension history length differs from precondition")
    if predecessor is not None:
        prior = existing[-1]
        if prior.get("extension_payload_sha256") != predecessor["extension_payload_sha256"]:
            raise RuntimeError("receipt predecessor payload differs from extension precondition")
        if prior.get("public_extension_commit") != predecessor["public_extension_commit"]:
            raise RuntimeError("receipt predecessor commit differs from extension precondition")
    if float(receipt.get("billing_cutoff_usd")) != budget["prior_public_receipt_billing_cutoff_usd"]:
        raise RuntimeError("receipt billing cutoff differs from extension precondition")
    if float(receipt.get("hard_total_cap_usd")) != budget["prior_public_receipt_hard_total_cap_usd"]:
        raise RuntimeError("receipt hard cap differs from extension precondition")
    if budget["new_hard_total_cap_usd"] - budget["new_billing_cutoff_usd"] != budget["non_compute_reserve_usd"]:
        raise RuntimeError("new budget does not preserve the frozen reserve")
    scientific = extension["scientific_protocol"]
    if any(scientific.values()):
        raise RuntimeError("budget extension changes or opens scientific state")
    return False


def apply(extension_path: Path, receipt_path: Path, public_commit: str) -> dict:
    extension = load_hashed(extension_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    already_applied = validate(extension, receipt)
    if already_applied:
        return receipt
    budget = extension["budget"]
    receipt.setdefault("budget_extensions", []).append(
        {
            "applied_at_utc": now(),
            "extension_file": extension_path.name,
            "extension_payload_sha256": extension["payload_sha256"],
            "public_extension_commit": require_commit(public_commit),
            "scientific_outputs_inspected_at_application": False,
        }
    )
    receipt["billing_cutoff_usd"] = budget["new_billing_cutoff_usd"]
    receipt["hard_total_cap_usd"] = budget["new_hard_total_cap_usd"]
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-extension-commit", required=True)
    args = parser.parse_args()
    receipt = apply(args.extension, args.receipt, args.public_extension_commit)
    print(
        json.dumps(
            {
                "billing_cutoff_usd": receipt["billing_cutoff_usd"],
                "hard_total_cap_usd": receipt["hard_total_cap_usd"],
                "plan_payload_sha256": receipt["plan_payload_sha256"],
                "scientific_outputs_inspected": receipt["scientific_outputs_inspected"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
