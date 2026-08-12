#!/usr/bin/env python3
"""Freeze seven Kaggle real and seven RunPod synthetic deliveries together."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import freeze_heldout_cache_delivery as freezer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    expected = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed is not None and observed != expected:
        raise RuntimeError("embedded payload SHA-256 mismatch")
    return expected


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--real-records", type=Path, required=True)
    parser.add_argument("--synthetic-records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("provider-neutral delivery output must start absent")
    public_plan_commit = freezer.require_hex(
        args.public_plan_commit, 40, "public execution-plan commit"
    )
    plan = load_hashed(args.plan)
    real_records = load_hashed(args.real_records)
    synthetic_records = load_hashed(args.synthetic_records)
    jobs = freezer.expected_jobs(plan)
    real_jobs, synthetic_jobs = jobs[:7], jobs[7:]
    public_plan_sha = sha256_file(args.plan)
    expected_public = {
        "commit": public_plan_commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": public_plan_sha,
        "payload_sha256": plan["payload_sha256"],
    }
    if real_records.get("public_plan") != expected_public:
        raise RuntimeError("real source records point to another public plan")
    if synthetic_records.get("public_execution_plan") != expected_public:
        raise RuntimeError("synthetic source records point to another public plan")
    if real_records.get("scientific_gate") != {
        "all_real_kernel_versions_match_acceptance_receipt": True,
        "all_real_kernel_sources_match_accepted_packages": True,
        "npz_probability_caches_downloaded": False,
        "kernel_logs_opened_or_read": False,
        "scientific_endpoints_opened_or_read": False,
    }:
        raise RuntimeError("real source-record blind gate mismatch")
    if synthetic_records.get("scientific_gate") != {
        "all_primary_jobs_complete": True,
        "all_copied_cache_hashes_verified": True,
        "npz_cache_payloads_opened_or_inspected": False,
        "scientific_endpoints_opened_or_read": False,
        "runpod_is_authoritative_primary": True,
    }:
        raise RuntimeError("RunPod source-record blind gate mismatch")
    real_sources = real_records.get("jobs")
    synthetic_sources = synthetic_records.get("jobs")
    if [item.get("job_id") for item in real_sources or []] != [
        job["job_id"] for job in real_jobs
    ]:
        raise RuntimeError("real source-record order mismatch")
    if [item.get("job_id") for item in synthetic_sources or []] != [
        job["job_id"] for job in synthetic_jobs
    ]:
        raise RuntimeError("synthetic source-record order mismatch")
    delivered = []
    for job, source in zip(real_jobs, real_sources, strict=True):
        index_path = Path(source["job_index"])
        identity = freezer.validate_job_index(
            path=index_path,
            job=job,
            plan=plan,
            public_plan_commit=public_plan_commit,
            public_plan_file_sha256=public_plan_sha,
        )
        origin = source.get("execution_origin")
        if (
            not isinstance(origin, dict)
            or set(origin) != {"provider", "kernel_id", "kernel_version"}
            or origin.get("provider") != "Kaggle"
            or not isinstance(origin.get("kernel_id"), str)
            or not origin["kernel_id"].startswith("aviadcohen1/")
            or not isinstance(origin.get("kernel_version"), int)
            or origin["kernel_version"] <= 0
        ):
            raise RuntimeError(f"{job['job_id']}: invalid Kaggle origin")
        delivered.append(
            {
                "job_id": job["job_id"],
                "mode": job["mode"],
                "run": job["run"],
                "kernel_id": origin["kernel_id"],
                "kernel_version": origin["kernel_version"],
                "execution_origin": origin,
                "job_index": identity,
            }
        )
    for job, source in zip(synthetic_jobs, synthetic_sources, strict=True):
        index_path = Path(source["job_index"])
        identity = freezer.validate_job_index(
            path=index_path,
            job=job,
            plan=plan,
            public_plan_commit=public_plan_commit,
            public_plan_file_sha256=public_plan_sha,
        )
        if source.get("job_index_identity") != identity:
            raise RuntimeError(f"{job['job_id']}: copied index identity mismatch")
        origin = source.get("execution_origin")
        if not isinstance(origin, dict) or origin.get("provider") != "RunPod":
            raise RuntimeError(f"{job['job_id']}: invalid RunPod origin")
        delivered.append(
            {
                "job_id": job["job_id"],
                "mode": job["mode"],
                "run": job["run"],
                "kernel_id": f"runpod:{origin['pod_id']}:{job['job_id']}",
                "kernel_version": 1,
                "execution_origin": origin,
                "job_index": identity,
            }
        )
    payload = {
        "schema_version": "1.0",
        "status": "all 14 publicly planned held-out caches sealed before one-shot scoring",
        "public_execution_plan": expected_public,
        "threshold_binding": plan["threshold_binding"],
        "delivery_source_records": {
            "real": {
                "file": args.real_records.name,
                "bytes": args.real_records.stat().st_size,
                "sha256": sha256_file(args.real_records),
                "payload_sha256": real_records["payload_sha256"],
            },
            "synthetic": {
                "file": args.synthetic_records.name,
                "bytes": args.synthetic_records.stat().st_size,
                "sha256": sha256_file(args.synthetic_records),
                "payload_sha256": synthetic_records["payload_sha256"],
            },
        },
        "job_order": [job["job_id"] for job in jobs],
        "jobs": delivered,
        "counts": {
            "jobs": 14,
            "real_jobs": 7,
            "synthetic_jobs": 7,
            "real_probability_caches": 266,
            "synthetic_ray_caches": 700,
        },
        "authority": {
            "real_provider": "Kaggle",
            "synthetic_primary_provider": "RunPod",
            "kaggle_synthetic_role": "sealed secondary cross-platform replication",
            "selection_between_synthetic_platforms_permitted": False,
        },
        "scientific_gate": {
            "all_cache_jobs_completed": True,
            "thresholds_publicly_frozen_before_cache_inference": True,
            "scientific_endpoints_scored": False,
            "scientific_endpoints_printed": False,
            "cache_npz_payloads_opened_or_inspected_by_delivery_freezer": False,
            "one_shot_scoring_permitted_after_this_public_freeze": True,
        },
        "delivery_freezer": {
            "file": Path(__file__).resolve().name,
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("ALL_PROVIDER_NEUTRAL_HELDOUT_CACHES_FROZEN_BEFORE_SCORING", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
