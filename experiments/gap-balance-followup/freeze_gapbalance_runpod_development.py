#!/usr/bin/env python3
"""Freeze the paid-provider execution amendment without touching results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
IMPORT_FIX = HERE / "GAPBALANCE_DEVELOPMENT_IMPORT_FIX.json"


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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-commit", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("RunPod plan output must start absent")
    if len(args.parent_commit) != 40 or any(c not in "0123456789abcdef" for c in args.parent_commit):
        raise RuntimeError("parent commit must be an exact lowercase Git SHA")
    fix = load_hashed(IMPORT_FIX)
    source_jobs = [job for job in fix["jobs"] if job["job_id"].startswith("gapbalance-development-synthetic-")]
    jobs = []
    for source in source_jobs:
        parts = source["job_id"].rsplit("-seed", 1)[1]
        seed_text, shard_text = parts.split("-shard", 1)
        job = dict(source)
        job["mode"] = "synthetic"
        job["seed"] = int(seed_text)
        job["shard_index"] = int(shard_text)
        jobs.append(job)
    if len(jobs) != 12:
        raise RuntimeError("import fix does not contain the exact 12 synthetic jobs")
    job_ids = [job["job_id"] for job in jobs]
    expected = [
        f"gapbalance-development-synthetic-seed{seed}-shard{shard:02d}"
        for seed in (11, 23, 47)
        for shard in range(4)
    ]
    if job_ids != expected:
        raise RuntimeError("synthetic job order differs from the public freeze")
    plan = {
        "schema_version": "1.0",
        "status": "RunPod development replacement frozen before paid execution or endpoint access",
        "public_parent_commit": args.parent_commit,
        "scientific_source_commit": fix["public_source_commit"],
        "development_import_fix_payload_sha256": fix["payload_sha256"],
        "private_package_manifest_payload_sha256": fix["v3_package_manifest_payload_sha256"],
        "authority": {
            "runpod_complete_12_job_set_is_primary": True,
            "kaggle_completed_jobs_are_secondary_replication_only": True,
            "cross_provider_partial_mix_for_selection_permitted": False,
            "partial_primary_scoring_or_selection_permitted": False,
        },
        "execution": {
            "provider": "RunPod community cloud",
            "existing_pod_only": True,
            "pod_id": "9r1cm9r82aonih",
            "pod_name": "vesuvius-synth-primary-ba4efa85-g7",
            "gpu_type": "RTX 4090",
            "gpu_count": 7,
            "container_image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "maximum_accepted_aggregate_hourly_rate_usd": 2.38,
            "waves": [job_ids[:7], job_ids[7:]],
        },
        "budget": {
            "user_authorization": "Aviad: ok go",
            "authorization_date": "2026-08-20",
            "absolute_campaign_cap_usd": 25.0,
            "development_billing_cutoff_usd": 12.0,
            "confirmation_reserve_usd": 13.0,
            "guard_seconds": 300,
            "on_cutoff": "stop the exact pod, preserve partial operational artifacts, do not score or select partial development output",
            "forms_authorized": False,
        },
        "sealed_gates": {
            "pherc1218_v2_opened": False,
            "confirmation_seeds_500_504_opened": False,
            "scientific_endpoints_scored": False,
            "confirmation_outputs_inspected": False,
            "holdout_access_before_public_candidate_and_threshold_freeze_permitted": False,
        },
        "jobs": jobs,
    }
    plan["payload_sha256"] = canonical(plan)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "jobs": len(jobs),
        "payload_sha256": plan["payload_sha256"],
        "development_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
