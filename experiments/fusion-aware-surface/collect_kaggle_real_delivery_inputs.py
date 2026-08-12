#!/usr/bin/env python3
"""Collect only seven sealed real-cache indexes/manifests from Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import collect_heldout_delivery_inputs as collector
import orchestrate_heldout_queue as queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--package-index", type=Path, required=True)
    parser.add_argument("--queue-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("real collector output must start absent")
    if len(args.public_plan_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in args.public_plan_commit
    ):
        raise ValueError("public plan commit must be 40 lowercase hexadecimal")
    plan = queue.load_canonical(args.plan)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong public held-out plan status")
    real_jobs = plan.get("real_test_jobs")
    if not isinstance(real_jobs, list) or len(real_jobs) != 7:
        raise RuntimeError("public plan does not contain seven real jobs")
    package_index, packages = queue.validate_packages(args.packages, args.package_index)
    receipt = queue.load_receipt(args.queue_receipt, args.package_index, package_index)
    accepted = queue.receipt_map(receipt, packages)
    real_job_ids = [job["job_id"] for job in real_jobs]
    if list(accepted)[:7] != real_job_ids or any(job_id not in accepted for job_id in real_job_ids):
        raise RuntimeError("queue receipt does not contain the seven real jobs in order")
    packages_by_job = {record["job_id"]: record for record in packages}
    args.out.mkdir(parents=True)
    downloads = args.out / "downloads"
    downloads.mkdir()
    sources = []
    for job in real_jobs:
        job_id = job["job_id"]
        record = accepted[job_id]
        package = packages_by_job[job_id]
        status = queue.kernel_status(args.kaggle, record["kernel_id"])
        if status != "COMPLETE":
            raise RuntimeError(f"{job_id}: real-cache kernel is not complete ({status})")
        version, source = collector.current_kernel_version_and_source(record["kernel_id"])
        if version != record["kernel_version"]:
            raise RuntimeError(f"{job_id}: latest Kaggle version changed")
        package_source = (
            args.packages
            / package["directory"]
            / plan["heldout_cache_launcher"]["file"]
        )
        if source != package_source.read_bytes():
            raise RuntimeError(f"{job_id}: Kaggle source differs from accepted package")
        destination = downloads / job_id
        collector.download_selected(args.kaggle, record["kernel_id"], destination)
        index_path = collector.validate_download(destination, job)
        sources.append(
            {
                "job_id": job_id,
                "execution_origin": {
                    "provider": "Kaggle",
                    "kernel_id": record["kernel_id"],
                    "kernel_version": version,
                },
                "job_index": str(index_path),
            }
        )
        print(f"KAGGLE_REAL_DELIVERY_INPUT_COLLECTED job={job_id}", flush=True)
    local = Path(__file__).resolve()
    payload = {
        "schema_version": "1.0",
        "status": "seven real held-out delivery indexes and manifests collected without NPZ",
        "public_plan": {
            "commit": args.public_plan_commit,
            "file": args.plan.name,
            "bytes": args.plan.stat().st_size,
            "sha256": queue.sha256_file(args.plan),
            "payload_sha256": plan["payload_sha256"],
        },
        "jobs": sources,
        "collector": {
            "file": local.name,
            "bytes": local.stat().st_size,
            "sha256": queue.sha256_file(local),
        },
        "scientific_gate": {
            "all_real_kernel_versions_match_acceptance_receipt": True,
            "all_real_kernel_sources_match_accepted_packages": True,
            "npz_probability_caches_downloaded": False,
            "kernel_logs_opened_or_read": False,
            "scientific_endpoints_opened_or_read": False,
        },
    }
    payload["payload_sha256"] = queue.canonical_sha256(payload)
    output = args.out / "kaggle_real_delivery_source_records.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("ALL_7_KAGGLE_REAL_DELIVERY_INPUTS_COLLECTED_WITHOUT_NPZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
