#!/usr/bin/env python3
"""Freeze the result-blind allocation retry after the exact host was full."""

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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("allocation retry output must start absent")
    if len(args.parent_commit) != 40 or any(c not in "0123456789abcdef" for c in args.parent_commit):
        raise RuntimeError("parent commit must be an exact lowercase Git SHA")
    plan = load_hashed(args.plan)
    retry = {
        "schema_version": "1.0",
        "status": "result-blind equivalent RunPod allocation retry frozen before creation",
        "public_parent_commit": args.parent_commit,
        "runpod_development_plan_payload_sha256": plan["payload_sha256"],
        "failed_zero_cost_attempt": {
            "pod_id": plan["execution"]["pod_id"],
            "provider_state_after_attempt": "EXITED",
            "runtime_after_attempt": None,
            "provider_error": "There are not enough free GPUs on the host machine to start this pod.",
            "allocation_started": False,
            "spend_usd": 0.0,
            "scientific_outputs_created_or_inspected": False,
        },
        "authorized_retry": {
            "route": "create one equivalent replacement allocation",
            "provider": "RunPod community cloud",
            "gpu_type_id": "NVIDIA GeForce RTX 4090",
            "gpu_count": 7,
            "container_image": plan["execution"]["container_image"],
            "maximum_accepted_aggregate_hourly_rate_usd": plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"],
            "volume_in_gb": 25,
            "container_disk_in_gb": 30,
            "public_ip_and_ssh": True,
            "job_set_or_wave_change": False,
        },
        "budget": plan["budget"],
        "authority": plan["authority"],
        "sealed_gates": plan["sealed_gates"],
    }
    retry["payload_sha256"] = canonical(retry)
    args.out.write_text(json.dumps(retry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "payload_sha256": retry["payload_sha256"],
        "failed_attempt_spend_usd": 0.0,
        "maximum_retry_hourly_rate_usd": retry["authorized_retry"]["maximum_accepted_aggregate_hourly_rate_usd"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
