#!/usr/bin/env python3
"""Freeze explicit bundle-egress approval and the exact stopped-pod resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError(f"payload SHA-256 mismatch: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--multi-pod-retry", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("egress-resume freeze output must start absent")
    if len(args.parent_commit) != 40 or any(c not in "0123456789abcdef" for c in args.parent_commit):
        raise RuntimeError("parent commit must be an exact lowercase Git SHA")
    plan = load_hashed(args.plan)
    multi = load_hashed(args.multi_pod_retry)
    bundle = load_hashed(args.bundle_manifest)
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "STOPPED"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("multi_pod_retry_payload_sha256") != multi["payload_sha256"]
        or bundle.get("plan_payload_sha256") != plan["payload_sha256"]
    ):
        raise RuntimeError("receipt, bundle, and public plans are not consistently bound")
    prior_spend = float(receipt["conservative_development_spend_usd"])
    if prior_spend < 0 or prior_spend >= float(plan["budget"]["development_billing_cutoff_usd"]):
        raise RuntimeError("prior conservative spend is outside the public cutoff")
    resume = {
        "schema_version": "1.0",
        "status": "explicit sealed-bundle egress and exact stopped-pod resume frozen before action",
        "public_parent_commit": args.parent_commit,
        "runpod_development_plan_payload_sha256": plan["payload_sha256"],
        "multi_pod_retry_payload_sha256": multi["payload_sha256"],
        "bundle_manifest_payload_sha256": bundle["payload_sha256"],
        "bundle_logical_bytes": sum(record["bytes"] for record in bundle["files"]),
        "user_authorization": "I APPROVE EVETHING YOU NEED WORK END TO END",
        "authorization_context": "explicit follow-up to the exact request to upload the 4.51 GB sealed bundle including private model weights, source/assets, and launchers to the three RunPod community-cloud pods",
        "egress": {
            "destination": "the exact three stopped RunPod community-cloud pods in the provider receipt",
            "replication_count": 3,
            "includes": ["private trained model weights", "source and frozen assets", "private generated launchers"],
            "pherc1218_included": False,
            "confirmation_seeds_500_504_included": False,
            "scientific_outputs_included": False,
        },
        "resume": {
            "exact_pods_only": [
                {"id": pod["id"], "gpu_count": pod["gpu_count"], "jobs": pod["jobs"]}
                for pod in receipt["pods"]
            ],
            "aggregate_hourly_rate_usd": receipt["total_hourly_rate_usd"],
            "prior_conservative_spend_usd": prior_spend,
            "remaining_development_cutoff_usd": round(
                float(plan["budget"]["development_billing_cutoff_usd"]) - prior_spend, 6
            ),
        },
        "budget": plan["budget"],
        "authority": plan["authority"],
        "sealed_gates": plan["sealed_gates"],
    }
    resume["payload_sha256"] = canonical(resume)
    args.out.write_text(json.dumps(resume, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "payload_sha256": resume["payload_sha256"],
        "prior_conservative_spend_usd": prior_spend,
        "remaining_development_cutoff_usd": resume["resume"]["remaining_development_cutoff_usd"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
