#!/usr/bin/env python3
"""Freeze the exact 14-job held-out cache delivery without opening NPZ payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    content = dict(payload)
    observed = content.pop("payload_sha256", None)
    expected = sha256_bytes(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed != expected:
        raise RuntimeError("embedded payload SHA-256 mismatch")
    return expected


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def require_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"invalid {label}")
    return value


def expected_kernel_id(job_id: str) -> str:
    return f"aviadcohen1/vesuvius-fusion-{job_id}"


def expected_job_config_identity(
    *, job_id: str, public_plan_commit: str, public_plan_file_sha256: str
) -> dict:
    config = {
        "schema_version": "1.0",
        "job_id": job_id,
        "public_plan_commit": public_plan_commit,
        "public_plan_file_sha256": public_plan_file_sha256,
    }
    config["payload_sha256"] = sha256_bytes(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    )
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "encoding": "hex in first source comment",
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "payload_sha256": config["payload_sha256"],
    }


def expected_jobs(plan: dict) -> list[dict]:
    real = plan.get("real_test_jobs")
    synthetic = plan.get("synthetic_ray_jobs")
    if not isinstance(real, list) or len(real) != 7:
        raise RuntimeError("public plan does not contain seven real cache jobs")
    if not isinstance(synthetic, list) or len(synthetic) != 7:
        raise RuntimeError("public plan does not contain seven synthetic cache jobs")
    jobs = real + synthetic
    if len({record.get("job_id") for record in jobs}) != 14:
        raise RuntimeError("public plan held-out job identities are not unique")
    return jobs


def validate_identity_record(
    record: dict, *, expected_count: int, mode: str
) -> None:
    base_keys = {"file", "bytes", "sha256"}
    if mode == "real_test_cache":
        expected_keys = base_keys | {"source_image"}
    elif mode == "synthetic_ray_cache":
        expected_keys = base_keys | {
            "name",
            "kind",
            "seed",
            "pitch_um",
            "papyrus",
            "kollesis",
        }
    else:
        raise RuntimeError("sealed cache-file mode is unsupported")
    if set(record) != expected_keys:
        raise RuntimeError("sealed cache-file identity schema mismatch")
    if (
        not isinstance(record["file"], str)
        or not record["file"].endswith(".npz")
        or not isinstance(record["bytes"], int)
        or record["bytes"] <= 0
    ):
        raise RuntimeError("sealed cache-file identity is invalid")
    require_hex(record["sha256"], 64, "sealed cache-file SHA-256")
    if mode == "real_test_cache":
        if not isinstance(record["source_image"], str) or not record["source_image"]:
            raise RuntimeError("real sealed cache-file source identity is invalid")
    else:
        if not isinstance(record["name"], str) or not record["name"]:
            raise RuntimeError("synthetic sealed cache-file name is invalid")
        if record["kind"] not in {"primary", "single_sheet_control"}:
            raise RuntimeError("synthetic sealed cache-file kind is invalid")
        if (
            not isinstance(record["seed"], int)
            or isinstance(record["seed"], bool)
            or not isinstance(record["pitch_um"], (int, float))
            or isinstance(record["pitch_um"], bool)
            or record["pitch_um"] <= 0
            or not isinstance(record["papyrus"], int)
            or isinstance(record["papyrus"], bool)
            or record["papyrus"] <= 0
            or not isinstance(record["kollesis"], bool)
        ):
            raise RuntimeError("synthetic sealed cache-file geometry is invalid")
    if expected_count <= 0:
        raise AssertionError("expected cache-file count must be positive")


def validate_job_index(
    *,
    path: Path,
    job: dict,
    plan: dict,
    public_plan_commit: str,
    public_plan_file_sha256: str,
) -> dict:
    index = load_json(path)
    payload_sha256 = canonical_payload_sha256(index)
    if index.get("schema_version") != "1.0":
        raise RuntimeError(f"{job['job_id']}: held-out index schema mismatch")
    if (
        index.get("status")
        != "one publicly planned held-out cache job sealed without scoring"
    ):
        raise RuntimeError(f"{job['job_id']}: held-out index status mismatch")
    if index.get("job") != job:
        raise RuntimeError(f"{job['job_id']}: held-out index job record mismatch")
    public = index.get("public_execution_plan", {})
    if public != {
        "commit": public_plan_commit,
        "file_sha256": public_plan_file_sha256,
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError(f"{job['job_id']}: public-plan binding mismatch")
    if index.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError(f"{job['job_id']}: threshold binding mismatch")
    if index.get("threshold_payload_sha256") != plan.get("threshold_binding", {}).get(
        "frozen_thresholds", {}
    ).get("payload_sha256"):
        raise RuntimeError(f"{job['job_id']}: threshold payload mismatch")
    if index.get("launcher") != plan.get("heldout_cache_launcher"):
        raise RuntimeError(f"{job['job_id']}: held-out launcher identity mismatch")
    if index.get("job_config") != expected_job_config_identity(
        job_id=job["job_id"],
        public_plan_commit=public_plan_commit,
        public_plan_file_sha256=public_plan_file_sha256,
    ):
        raise RuntimeError(f"{job['job_id']}: embedded job-config identity mismatch")
    gate = index.get("scientific_gate", {})
    expected_gate = {
        "thresholds_were_publicly_frozen_before_this_job": True,
        "held_out_inference_executed": True,
        "scientific_endpoints_scored": False,
        "scientific_endpoints_printed": False,
        "manual_threshold_override_used": False,
        "cache_payload_opened_or_inspected_by_launcher": False,
    }
    if gate != expected_gate:
        raise RuntimeError(f"{job['job_id']}: held-out blind gate mismatch")

    mode = job.get("mode")
    expected_manifest_count = 1 if mode == "real_test_cache" else 10
    expected_file_count = 38 if mode == "real_test_cache" else 100
    manifest_records = index.get("cache_manifests")
    sealed_files = index.get("sealed_cache_files")
    if (
        not isinstance(manifest_records, list)
        or len(manifest_records) != expected_manifest_count
    ):
        raise RuntimeError(f"{job['job_id']}: cache-manifest count mismatch")
    if not isinstance(sealed_files, list) or len(sealed_files) != expected_file_count:
        raise RuntimeError(f"{job['job_id']}: sealed cache-file count mismatch")
    expected_manifest_names = (
        {job["expected_cache_manifest"]}
        if mode == "real_test_cache"
        else set(job["expected_cache_manifests"])
    )
    observed_manifest_names = set()
    for record in manifest_records:
        if set(record) != {"file", "bytes", "sha256", "payload_sha256"}:
            raise RuntimeError(
                f"{job['job_id']}: cache-manifest identity schema mismatch"
            )
        observed_manifest_names.add(record["file"])
        if not isinstance(record["bytes"], int) or record["bytes"] <= 0:
            raise RuntimeError(f"{job['job_id']}: cache-manifest byte size is invalid")
        require_hex(record["sha256"], 64, "cache-manifest SHA-256")
        require_hex(record["payload_sha256"], 64, "cache-manifest payload SHA-256")
    if observed_manifest_names != expected_manifest_names:
        raise RuntimeError(f"{job['job_id']}: cache-manifest file set mismatch")
    for record in sealed_files:
        validate_identity_record(
            record, expected_count=expected_file_count, mode=mode
        )
    names = [record["file"] for record in sealed_files]
    if len(set(names)) != expected_file_count:
        raise RuntimeError(f"{job['job_id']}: duplicate sealed cache-file identity")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload_sha256": payload_sha256,
        "cache_manifest_count": expected_manifest_count,
        "sealed_cache_file_count": expected_file_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"output must start absent: {args.out}")
    public_plan_commit = require_hex(
        args.public_plan_commit, 40, "public execution-plan commit"
    )
    public_plan_file_sha256 = sha256_file(args.plan)
    plan = load_json(args.plan)
    canonical_payload_sha256(plan)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong public execution-plan status")
    local_freezer = {
        "file": Path(__file__).resolve().name,
        "bytes": Path(__file__).resolve().stat().st_size,
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    if plan.get("heldout_delivery_freezer") != local_freezer:
        raise RuntimeError("running delivery freezer differs from public plan")
    jobs = expected_jobs(plan)
    records = load_json(args.records)
    records_payload_sha256 = canonical_payload_sha256(records)
    if records.get("schema_version") != "1.0":
        raise RuntimeError("delivery-source record schema mismatch")
    if records.get("status") != (
        "all held-out delivery indexes and manifests collected without NPZ"
    ):
        raise RuntimeError("delivery-source record status mismatch")
    if records.get("public_plan") != {
        "commit": public_plan_commit,
        "file": args.plan.name,
        "bytes": args.plan.stat().st_size,
        "sha256": public_plan_file_sha256,
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("delivery-source record public-plan binding mismatch")
    if records.get("collector") != plan.get("heldout_delivery_collector"):
        raise RuntimeError("delivery-source collector identity mismatch")
    if records.get("scientific_gate") != {
        "all_latest_kernel_versions_match_acceptance_receipt": True,
        "all_kernel_sources_match_accepted_packages": True,
        "npz_probability_caches_downloaded": False,
        "kernel_logs_opened_or_read": False,
        "scientific_endpoints_opened_or_read": False,
    }:
        raise RuntimeError("delivery-source blind gate mismatch")
    sources = records.get("jobs")
    if not isinstance(sources, list) or [item.get("job_id") for item in sources] != [
        job["job_id"] for job in jobs
    ]:
        raise RuntimeError("delivery-source records do not follow public plan order")

    delivery = []
    for job, source in zip(jobs, sources, strict=True):
        if set(source) != {"job_id", "kernel_id", "kernel_version", "job_index"}:
            raise RuntimeError(f"{job['job_id']}: delivery-source schema mismatch")
        if source["job_id"] != job["job_id"]:
            raise RuntimeError("delivery-source job identity mismatch")
        if source["kernel_id"] != expected_kernel_id(job["job_id"]):
            raise RuntimeError(f"{job['job_id']}: Kaggle kernel identity mismatch")
        if (
            not isinstance(source["kernel_version"], int)
            or source["kernel_version"] <= 0
        ):
            raise RuntimeError(f"{job['job_id']}: Kaggle kernel version is invalid")
        index_path = Path(source["job_index"])
        if not index_path.is_file() or index_path.name != "heldout_job_index.json":
            raise RuntimeError(f"{job['job_id']}: held-out job index is absent")
        index_identity = validate_job_index(
            path=index_path,
            job=job,
            plan=plan,
            public_plan_commit=public_plan_commit,
            public_plan_file_sha256=public_plan_file_sha256,
        )
        delivery.append(
            {
                "job_id": job["job_id"],
                "mode": job["mode"],
                "run": job["run"],
                "kernel_id": source["kernel_id"],
                "kernel_version": source["kernel_version"],
                "job_index": index_identity,
            }
        )

    payload = {
        "schema_version": "1.0",
        "status": "all 14 publicly planned held-out caches sealed before one-shot scoring",
        "public_execution_plan": {
            "commit": public_plan_commit,
            "file": args.plan.name,
            "bytes": args.plan.stat().st_size,
            "sha256": public_plan_file_sha256,
            "payload_sha256": plan["payload_sha256"],
        },
        "threshold_binding": plan["threshold_binding"],
        "delivery_source_records": {
            "file": args.records.name,
            "bytes": args.records.stat().st_size,
            "sha256": sha256_file(args.records),
            "payload_sha256": records_payload_sha256,
        },
        "job_order": [job["job_id"] for job in jobs],
        "jobs": delivery,
        "counts": {
            "jobs": 14,
            "real_jobs": 7,
            "synthetic_jobs": 7,
            "real_probability_caches": 266,
            "synthetic_ray_caches": 700,
        },
        "scientific_gate": {
            "all_cache_jobs_completed": True,
            "thresholds_publicly_frozen_before_cache_inference": True,
            "scientific_endpoints_scored": False,
            "scientific_endpoints_printed": False,
            "cache_npz_payloads_opened_or_inspected_by_delivery_freezer": False,
            "one_shot_scoring_permitted_after_this_public_freeze": True,
        },
        "delivery_freezer": local_freezer,
    }
    payload["payload_sha256"] = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("held-out delivery jobs:", len(delivery))
    print("held-out delivery payload SHA-256:", payload["payload_sha256"])
    print("ALL_HELDOUT_CACHES_FROZEN_BEFORE_ONE_SHOT_SCORING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
