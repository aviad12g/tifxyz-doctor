#!/usr/bin/env python3
"""Validate copied RunPod primary outputs without opening any NPZ payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import freeze_heldout_cache_delivery as delivery


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
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    if payload.get("payload_sha256") != canonical_sha256(payload):
        raise RuntimeError(f"embedded payload SHA-256 mismatch: {path}")
    return payload


def require_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {label}")
    return value


def synthetic_jobs(plan: dict) -> list[dict]:
    jobs = plan.get("synthetic_ray_jobs")
    if not isinstance(jobs, list) or len(jobs) != 7:
        raise RuntimeError("held-out plan does not contain seven synthetic jobs")
    if len({job.get("job_id") for job in jobs}) != 7:
        raise RuntimeError("synthetic job identities are not unique")
    return jobs


def validate_complete_status(status: dict, job_ids: list[str], plan_sha: str) -> None:
    if status.get("state") != "COMPLETE":
        raise RuntimeError("RunPod primary status is not COMPLETE")
    if status.get("scientific_outputs_inspected") is not False:
        raise RuntimeError("RunPod primary status does not preserve the blind gate")
    if status.get("plan_payload_sha256") != plan_sha:
        raise RuntimeError("RunPod status points to another replacement plan")
    jobs = status.get("jobs")
    if not isinstance(jobs, dict) or list(jobs) != job_ids:
        raise RuntimeError("RunPod status job order mismatch")
    for job_id in job_ids:
        record = jobs[job_id]
        if record.get("state") != "COMPLETE" or record.get("returncode") != 0:
            raise RuntimeError(f"{job_id}: RunPod job is not complete")
        if not isinstance(record.get("gpu_index"), int):
            raise RuntimeError(f"{job_id}: missing GPU isolation record")


def verify_job_tree(
    *,
    root: Path,
    job: dict,
    heldout_plan: dict,
    public_plan_commit: str,
    public_plan_sha: str,
) -> tuple[dict, int, int]:
    job_id = job["job_id"]
    job_root = root / job_id
    output_root = job_root / f"fusion-aware-heldout-{job_id}"
    index_path = output_root / "heldout_job_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"{job_id}: held-out job index is absent")
    if {path.name for path in job_root.iterdir()} != {output_root.name}:
        raise RuntimeError(f"{job_id}: unexpected worker output file set")
    index_identity = delivery.validate_job_index(
        path=index_path,
        job=job,
        plan=heldout_plan,
        public_plan_commit=public_plan_commit,
        public_plan_file_sha256=public_plan_sha,
    )
    index = load_hashed(index_path)
    manifest_records = index["cache_manifests"]
    cache_records = index["sealed_cache_files"]
    expected_root_files = {"heldout_job_index.json"} | {
        record["file"] for record in manifest_records
    }
    observed_root_files = {
        path.name for path in output_root.iterdir() if path.is_file()
    }
    observed_root_dirs = {
        path.name for path in output_root.iterdir() if path.is_dir()
    }
    if observed_root_files != expected_root_files or observed_root_dirs != {job["run"]}:
        raise RuntimeError(f"{job_id}: sealed output-root file set mismatch")
    for record in manifest_records:
        path = output_root / record["file"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"{job_id}: cache-manifest identity mismatch")
        manifest = load_hashed(path)
        if manifest["payload_sha256"] != record["payload_sha256"]:
            raise RuntimeError(f"{job_id}: cache-manifest payload mismatch")
    run_root = output_root / job["run"]
    observed_cache_names = {
        path.name for path in run_root.iterdir() if path.is_file()
    }
    expected_cache_names = {record["file"] for record in cache_records}
    if observed_cache_names != expected_cache_names:
        raise RuntimeError(f"{job_id}: sealed cache file set mismatch")
    for record in cache_records:
        path = run_root / record["file"]
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"{job_id}: sealed cache identity mismatch: {record['file']}")
    return index_identity, len(manifest_records), len(cache_records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--heldout-plan", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--replacement-plan", type=Path, required=True)
    parser.add_argument("--public-replacement-commit", required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("collector output must start absent")
    public_plan_commit = require_hex(args.public_plan_commit, 40, "public plan commit")
    public_replacement_commit = require_hex(
        args.public_replacement_commit, 40, "public replacement commit"
    )
    heldout_plan = load_hashed(args.heldout_plan)
    replacement = load_hashed(args.replacement_plan)
    if heldout_plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong held-out execution plan status")
    if replacement.get("execution", {}).get("runpod_is_authoritative_primary") is not True:
        raise RuntimeError("replacement plan does not declare RunPod primary authority")
    jobs = synthetic_jobs(heldout_plan)
    job_ids = [job["job_id"] for job in jobs]
    if [job.get("job_id") for job in replacement.get("jobs", [])] != job_ids:
        raise RuntimeError("replacement-plan job order mismatch")
    status = json.loads(args.status.read_text(encoding="utf-8"))
    validate_complete_status(status, job_ids, replacement["payload_sha256"])
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("plan_payload_sha256") != replacement["payload_sha256"]
        or receipt.get("public_replacement_commit") != public_replacement_commit
        or receipt.get("scientific_outputs_inspected") is not False
    ):
        raise RuntimeError("provider receipt binding mismatch")
    pods = receipt.get("pods")
    if not isinstance(pods, list) or len(pods) != 1 or pods[0].get("jobs") != job_ids:
        raise RuntimeError("provider receipt does not bind one exact seven-job pod")
    observed_jobs = {path.name for path in args.root.iterdir() if path.is_dir()}
    observed_files = {path.name for path in args.root.iterdir() if path.is_file()}
    if observed_jobs != set(job_ids) or observed_files:
        raise RuntimeError("copied RunPod output top-level file set mismatch")
    public_plan_sha = sha256_file(args.heldout_plan)
    sources = []
    manifest_count = cache_count = 0
    for job in jobs:
        identity, manifests, caches = verify_job_tree(
            root=args.root,
            job=job,
            heldout_plan=heldout_plan,
            public_plan_commit=public_plan_commit,
            public_plan_sha=public_plan_sha,
        )
        manifest_count += manifests
        cache_count += caches
        sources.append(
            {
                "job_id": job["job_id"],
                "execution_origin": {
                    "provider": "RunPod",
                    "pod_id": pods[0]["id"],
                    "public_replacement_commit": public_replacement_commit,
                    "replacement_plan_payload_sha256": replacement["payload_sha256"],
                },
                "job_index": str(
                    args.root
                    / job["job_id"]
                    / f"fusion-aware-heldout-{job['job_id']}"
                    / "heldout_job_index.json"
                ),
                "job_index_identity": identity,
            }
        )
        print(f"RUNPOD_PRIMARY_JOB_VERIFIED job={job['job_id']}", flush=True)
    if (manifest_count, cache_count) != (70, 700):
        raise RuntimeError("RunPod primary aggregate count mismatch")
    collector = Path(__file__).resolve()
    payload = {
        "schema_version": "1.0",
        "status": "seven authoritative RunPod primary deliveries verified without opening NPZ",
        "public_execution_plan": {
            "commit": public_plan_commit,
            "file": args.heldout_plan.name,
            "bytes": args.heldout_plan.stat().st_size,
            "sha256": public_plan_sha,
            "payload_sha256": heldout_plan["payload_sha256"],
        },
        "runpod_replacement": {
            "commit": public_replacement_commit,
            "file": args.replacement_plan.name,
            "bytes": args.replacement_plan.stat().st_size,
            "sha256": sha256_file(args.replacement_plan),
            "payload_sha256": replacement["payload_sha256"],
        },
        "provider_receipt": {
            "file": args.provider_receipt.name,
            "bytes": args.provider_receipt.stat().st_size,
            "sha256": sha256_file(args.provider_receipt),
        },
        "jobs": sources,
        "counts": {"jobs": 7, "cache_manifests": 70, "sealed_cache_files": 700},
        "collector": {
            "file": collector.name,
            "bytes": collector.stat().st_size,
            "sha256": sha256_file(collector),
        },
        "scientific_gate": {
            "all_primary_jobs_complete": True,
            "all_copied_cache_hashes_verified": True,
            "npz_cache_payloads_opened_or_inspected": False,
            "scientific_endpoints_opened_or_read": False,
            "runpod_is_authoritative_primary": True,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("ALL_7_RUNPOD_PRIMARY_DELIVERIES_VERIFIED_WITHOUT_NPZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
