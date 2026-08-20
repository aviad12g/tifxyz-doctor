#!/usr/bin/env python3
"""Freeze equivalent multi-pod layouts after two seven-GPU hosts vanished."""

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


def partitions(job_ids: list[str], gpu_counts: list[int], job_counts: list[int]) -> list[dict]:
    if sum(job_counts) != len(job_ids) or len(gpu_counts) != len(job_counts):
        raise RuntimeError("invalid multi-pod partition shape")
    result = []
    cursor = 0
    for gpu_count, job_count in zip(gpu_counts, job_counts):
        jobs = job_ids[cursor:cursor + job_count]
        cursor += job_count
        result.append({
            "gpu_count": gpu_count,
            "jobs": jobs,
            "waves": [jobs[index:index + gpu_count] for index in range(0, len(jobs), gpu_count)],
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--allocation-retry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("multi-pod retry output must start absent")
    if len(args.parent_commit) != 40 or any(c not in "0123456789abcdef" for c in args.parent_commit):
        raise RuntimeError("parent commit must be an exact lowercase Git SHA")
    plan = load_hashed(args.plan)
    prior = load_hashed(args.allocation_retry)
    if prior["runpod_development_plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("prior retry is bound to another plan")
    jobs = [job["job_id"] for job in plan["jobs"]]
    layouts = [
        {"name": "four_plus_three", "partitions": partitions(jobs, [4, 3], [7, 5])},
        {"name": "three_plus_two_plus_two", "partitions": partitions(jobs, [3, 2, 2], [5, 4, 3])},
        {"name": "two_plus_two_plus_two_plus_one", "partitions": partitions(jobs, [2, 2, 2, 1], [4, 3, 3, 2])},
        {"name": "seven_singles", "partitions": partitions(jobs, [1] * 7, [2, 2, 2, 2, 2, 1, 1])},
    ]
    retry = {
        "schema_version": "1.0",
        "status": "result-blind equivalent multi-pod RunPod retry frozen before creation",
        "public_parent_commit": args.parent_commit,
        "runpod_development_plan_payload_sha256": plan["payload_sha256"],
        "prior_allocation_retry_payload_sha256": prior["payload_sha256"],
        "failed_zero_cost_attempt": {
            "route": "one new seven-GPU community pod",
            "provider_error": "There are no longer any instances available with the requested specifications. Please refresh and try again.",
            "allocation_started": False,
            "spend_usd": 0.0,
            "scientific_outputs_created_or_inspected": False,
        },
        "provider_contract": {
            "provider": "RunPod community cloud",
            "gpu_type_id": "NVIDIA GeForce RTX 4090",
            "total_gpu_count": 7,
            "per_gpu_hourly_ceiling_usd": 0.34,
            "aggregate_hourly_ceiling_usd": 2.38,
            "container_image": plan["execution"]["container_image"],
            "volume_in_gb_per_pod": 25,
            "container_disk_in_gb_per_pod": 30,
        },
        "allowed_layouts_in_order": layouts,
        "budget": plan["budget"],
        "authority": plan["authority"],
        "sealed_gates": plan["sealed_gates"],
    }
    retry["payload_sha256"] = canonical(retry)
    args.out.write_text(json.dumps(retry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "payload_sha256": retry["payload_sha256"],
        "layouts": [layout["name"] for layout in layouts],
        "aggregate_hourly_ceiling_usd": 2.38,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
