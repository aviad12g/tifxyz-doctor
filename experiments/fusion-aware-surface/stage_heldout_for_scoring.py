#!/usr/bin/env python3
"""Verify a public held-out delivery and stage caches without opening NPZ data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

RUN_ORDER = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
OPERATIONAL_PLAN_FIELDS = {
    "payload_sha256",
    "one_shot_scoring_launcher",
    "one_shot_scoring_stager",
    "scoring_package_generator",
    "scoring_pair_controller",
    "final_result_validator",
    "result_blind_scoring_asset_correction",
    "result_blind_scoring_job_plan_compatibility_correction",
    "result_blind_scoring_cache_identity_schema_correction",
    "result_blind_scoring_runtime_transport_correction",
    "result_blind_scoring_runtime_projection_correction",
    "result_blind_parallel_real_scoring_retry",
    "result_blind_parallel_real_asset_transport",
    "result_blind_runpod_real_scoring_retry",
    "result_blind_runpod_cpu_real_scoring",
    "result_blind_runpod_cpu_projection_correction",
    "result_blind_runpod_cpu_exact_scorer_correction",
    "runpod_cpu_scoring_asset_stager",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    observed = payload.get("payload_sha256")
    content = dict(payload)
    content.pop("payload_sha256", None)
    if observed != canonical_sha256(content):
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


def jobs_for_mode(plan: dict, mode: str) -> list[dict]:
    key = "real_test_jobs" if mode == "real" else "synthetic_ray_jobs"
    jobs = plan.get(key)
    if not isinstance(jobs, list) or len(jobs) != 7:
        raise RuntimeError(f"public plan does not contain seven {mode} jobs")
    if [job.get("run") for job in jobs] != list(RUN_ORDER):
        raise RuntimeError(f"public {mode} job order mismatch")
    expected_mode = "real_test_cache" if mode == "real" else "synthetic_ray_cache"
    if any(job.get("mode") != expected_mode for job in jobs):
        raise RuntimeError(f"public {mode} job type mismatch")
    return jobs


def validate_plan_delivery(
    plan_path: Path,
    cache_job_plan_path: Path,
    delivery_path: Path,
    public_plan_commit: str,
    public_delivery_commit: str,
) -> tuple[dict, dict, dict]:
    require_hex(public_plan_commit, 40, "public plan commit")
    require_hex(public_delivery_commit, 40, "public delivery commit")
    plan = load_hashed(plan_path)
    cache_job_plan = load_hashed(cache_job_plan_path)
    delivery = load_hashed(delivery_path)
    if plan.get("status") != "held-out execution plan frozen before held-out inference":
        raise RuntimeError("wrong public held-out plan status")
    local_stager = {
        "file": Path(__file__).resolve().name,
        "bytes": Path(__file__).resolve().stat().st_size,
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    if plan.get("one_shot_scoring_stager") != local_stager:
        raise RuntimeError("running scoring stager differs from public plan")
    correction = plan.get("result_blind_scoring_asset_correction", {})
    cache_job_identity = correction.get("predecessor_public_execution_plan")
    if cache_job_identity != {
        "commit": cache_job_identity.get("commit") if isinstance(cache_job_identity, dict) else None,
        "file": "heldout_execution_plan.json",
        "bytes": cache_job_plan_path.stat().st_size,
        "sha256": sha256_file(cache_job_plan_path),
        "payload_sha256": cache_job_plan["payload_sha256"],
    }:
        raise RuntimeError("cache-job execution plan differs from public correction")
    require_hex(cache_job_identity["commit"], 40, "cache-job plan commit")
    current_scientific = {
        key: value for key, value in plan.items() if key not in OPERATIONAL_PLAN_FIELDS
    }
    cache_job_scientific = {
        key: value
        for key, value in cache_job_plan.items()
        if key not in OPERATIONAL_PLAN_FIELDS
    }
    if current_scientific != cache_job_scientific:
        raise RuntimeError("corrected plan changes the cache-job scientific contract")
    if delivery.get("status") != (
        "all 14 publicly planned held-out caches sealed before one-shot scoring"
    ):
        raise RuntimeError("wrong public held-out delivery status")
    binding = delivery.get("public_execution_plan", {})
    if binding != {
        "commit": public_plan_commit,
        "file": plan_path.name,
        "bytes": plan_path.stat().st_size,
        "sha256": sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }:
        raise RuntimeError("held-out delivery points to another public plan")
    if delivery.get("threshold_binding") != plan.get("threshold_binding"):
        raise RuntimeError("held-out delivery threshold binding mismatch")
    expected_jobs = jobs_for_mode(plan, "real") + jobs_for_mode(plan, "synthetic")
    if delivery.get("job_order") != [job["job_id"] for job in expected_jobs]:
        raise RuntimeError("held-out delivery job order mismatch")
    delivered = delivery.get("jobs")
    if not isinstance(delivered, list) or len(delivered) != 14:
        raise RuntimeError("held-out delivery does not contain 14 jobs")
    if [record.get("job_id") for record in delivered] != delivery["job_order"]:
        raise RuntimeError("held-out delivery records are out of order")
    if delivery.get("counts") != {
        "jobs": 14,
        "real_jobs": 7,
        "synthetic_jobs": 7,
        "real_probability_caches": 266,
        "synthetic_ray_caches": 700,
    }:
        raise RuntimeError("held-out delivery counts mismatch")
    if delivery.get("scientific_gate") != {
        "all_cache_jobs_completed": True,
        "thresholds_publicly_frozen_before_cache_inference": True,
        "scientific_endpoints_scored": False,
        "scientific_endpoints_printed": False,
        "cache_npz_payloads_opened_or_inspected_by_delivery_freezer": False,
        "one_shot_scoring_permitted_after_this_public_freeze": True,
    }:
        raise RuntimeError("held-out delivery blind gate mismatch")
    return plan, delivery, cache_job_plan


def find_exact_index(input_root: Path, identity: dict) -> Path:
    expected = {
        "file",
        "bytes",
        "sha256",
        "payload_sha256",
        "cache_manifest_count",
        "sealed_cache_file_count",
    }
    if set(identity) != expected or identity.get("file") != "heldout_job_index.json":
        raise RuntimeError("held-out job-index identity schema mismatch")
    matches = [
        path.resolve()
        for path in input_root.rglob("heldout_job_index.json")
        if path.is_file()
        and path.stat().st_size == identity["bytes"]
        and sha256_file(path) == identity["sha256"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact held-out job index; found {matches}")
    return matches[0]


def validate_cache_identity_record(record: dict, mode: str, job_id: str) -> None:
    base = {"file", "bytes", "sha256"}
    expected = (
        base | {"source_image"}
        if mode == "real_test_cache"
        else base
        | {"name", "kind", "seed", "pitch_um", "papyrus", "kollesis"}
    )
    if set(record) != expected:
        raise RuntimeError(f"{job_id}: cache identity schema mismatch")
    if (
        not isinstance(record["file"], str)
        or not record["file"].endswith(".npz")
        or not isinstance(record["bytes"], int)
        or isinstance(record["bytes"], bool)
        or record["bytes"] <= 0
    ):
        raise RuntimeError(f"{job_id}: cache identity is invalid")
    require_hex(record["sha256"], 64, "cache SHA-256")
    if mode == "real_test_cache":
        if not isinstance(record["source_image"], str) or not record["source_image"]:
            raise RuntimeError(f"{job_id}: real cache source identity is invalid")
        return
    if (
        not isinstance(record["name"], str)
        or not record["name"]
        or record["kind"] not in {"primary", "single_sheet_control"}
        or not isinstance(record["seed"], int)
        or isinstance(record["seed"], bool)
        or not isinstance(record["pitch_um"], (int, float))
        or isinstance(record["pitch_um"], bool)
        or record["pitch_um"] <= 0
        or not isinstance(record["papyrus"], int)
        or isinstance(record["papyrus"], bool)
        or record["papyrus"] <= 0
        or not isinstance(record["kollesis"], bool)
    ):
        raise RuntimeError(f"{job_id}: synthetic cache source identity is invalid")


def verify_job_root(
    *,
    index_path: Path,
    identity: dict,
    job: dict,
    plan: dict,
    cache_job_plan: dict,
    cache_job_plan_commit: str,
    cache_job_plan_path: Path,
) -> tuple[dict, list[dict], list[dict]]:
    index = load_hashed(index_path)
    if index.get("payload_sha256") != identity.get("payload_sha256"):
        raise RuntimeError(f"{job['job_id']}: job-index payload mismatch")
    if index.get("status") != (
        "one publicly planned held-out cache job sealed without scoring"
    ):
        raise RuntimeError(f"{job['job_id']}: wrong job-index status")
    if index.get("job") != job:
        raise RuntimeError(f"{job['job_id']}: job-index job record mismatch")
    if index.get("public_execution_plan") != {
        "commit": cache_job_plan_commit,
        "file_sha256": sha256_file(cache_job_plan_path),
        "payload_sha256": cache_job_plan["payload_sha256"],
    }:
        raise RuntimeError(f"{job['job_id']}: job-index plan binding mismatch")
    if index.get("threshold_binding") != cache_job_plan.get("threshold_binding"):
        raise RuntimeError(f"{job['job_id']}: job-index threshold binding mismatch")
    if index.get("launcher") != cache_job_plan.get("heldout_cache_launcher"):
        raise RuntimeError(f"{job['job_id']}: job-index launcher mismatch")
    if index.get("scientific_gate") != {
        "thresholds_were_publicly_frozen_before_this_job": True,
        "held_out_inference_executed": True,
        "scientific_endpoints_scored": False,
        "scientific_endpoints_printed": False,
        "manual_threshold_override_used": False,
        "cache_payload_opened_or_inspected_by_launcher": False,
    }:
        raise RuntimeError(f"{job['job_id']}: job-index blind gate mismatch")
    manifests = index.get("cache_manifests")
    caches = index.get("sealed_cache_files")
    expected_manifest_count = 1 if job["mode"] == "real_test_cache" else 10
    expected_cache_count = 38 if job["mode"] == "real_test_cache" else 100
    if not isinstance(manifests, list) or len(manifests) != expected_manifest_count:
        raise RuntimeError(f"{job['job_id']}: manifest count mismatch")
    if not isinstance(caches, list) or len(caches) != expected_cache_count:
        raise RuntimeError(f"{job['job_id']}: cache count mismatch")
    root = index_path.parent
    for record in manifests:
        if set(record) != {"file", "bytes", "sha256", "payload_sha256"}:
            raise RuntimeError(f"{job['job_id']}: manifest identity schema mismatch")
        path = root / record["file"]
        manifest = load_hashed(path)
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
            or manifest["payload_sha256"] != record["payload_sha256"]
        ):
            raise RuntimeError(f"{job['job_id']}: manifest identity mismatch")
    expected_names = set()
    for record in caches:
        validate_cache_identity_record(record, job["mode"], job["job_id"])
        name = record["file"]
        if (
            not isinstance(name, str)
            or not name.endswith(".npz")
            or name in expected_names
        ):
            raise RuntimeError(f"{job['job_id']}: invalid or duplicate cache name")
        expected_names.add(name)
        path = root / job["run"] / name
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"{job['job_id']}: cache identity mismatch for {name}")
    observed_names = {
        path.name for path in (root / job["run"]).iterdir() if path.is_file()
    }
    if observed_names != expected_names:
        raise RuntimeError(f"{job['job_id']}: mounted run-directory mismatch")
    return index, manifests, caches


def relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, destination.parent))


def stage_mode(
    *,
    mode: str,
    input_root: Path,
    output_root: Path,
    plan_path: Path,
    cache_job_plan_path: Path,
    plan: dict,
    cache_job_plan: dict,
    delivery: dict,
    public_plan_commit: str,
    public_delivery_commit: str,
) -> dict:
    jobs = jobs_for_mode(plan, mode)
    delivered = {record["job_id"]: record for record in delivery["jobs"]}
    output_root.mkdir(parents=True, exist_ok=False)
    staged = []
    total_manifests = 0
    total_caches = 0
    for job in jobs:
        delivery_record = delivered.get(job["job_id"])
        if not isinstance(delivery_record, dict):
            raise TypeError(f"{job['job_id']}: absent from public delivery")
        if (
            delivery_record.get("mode") != job["mode"]
            or delivery_record.get("run") != job["run"]
        ):
            raise RuntimeError(f"{job['job_id']}: delivery identity mismatch")
        index_identity = delivery_record.get("job_index")
        index_path = find_exact_index(input_root, index_identity)
        _, manifests, caches = verify_job_root(
            index_path=index_path,
            identity=index_identity,
            job=job,
            plan=plan,
            cache_job_plan=cache_job_plan,
            cache_job_plan_commit=plan[
                "result_blind_scoring_asset_correction"
            ]["predecessor_public_execution_plan"]["commit"],
            cache_job_plan_path=cache_job_plan_path,
        )
        source_root = index_path.parent
        for manifest in manifests:
            relative_symlink(
                source_root / manifest["file"], output_root / manifest["file"]
            )
        relative_symlink(source_root / job["run"], output_root / job["run"])
        total_manifests += len(manifests)
        total_caches += len(caches)
        staged.append(
            {
                "job_id": job["job_id"],
                "run": job["run"],
                "kernel_id": delivery_record["kernel_id"],
                "kernel_version": delivery_record["kernel_version"],
                "job_index": index_identity,
                "cache_manifest_count": len(manifests),
                "sealed_cache_file_count": len(caches),
            }
        )
    expected = (7, 266) if mode == "real" else (70, 700)
    if (total_manifests, total_caches) != expected:
        raise RuntimeError(
            f"{mode}: staged count mismatch: {(total_manifests, total_caches)}"
        )
    payload = {
        "schema_version": "1.0",
        "status": f"{mode} held-out caches staged from public delivery before one-shot scoring",
        "mode": mode,
        "public_execution_plan": {
            "commit": public_plan_commit,
            "file": plan_path.name,
            "bytes": plan_path.stat().st_size,
            "sha256": sha256_file(plan_path),
            "payload_sha256": plan["payload_sha256"],
        },
        "public_cache_delivery": {
            "commit": public_delivery_commit,
            "payload_sha256": delivery["payload_sha256"],
        },
        "cache_job_execution_plan": {
            "commit": plan["result_blind_scoring_asset_correction"][
                "predecessor_public_execution_plan"
            ]["commit"],
            "file": cache_job_plan_path.name,
            "bytes": cache_job_plan_path.stat().st_size,
            "sha256": sha256_file(cache_job_plan_path),
            "payload_sha256": cache_job_plan["payload_sha256"],
        },
        "threshold_binding": plan["threshold_binding"],
        "job_order": [job["job_id"] for job in jobs],
        "jobs": staged,
        "counts": {
            "jobs": 7,
            "cache_manifests": total_manifests,
            "sealed_cache_files": total_caches,
        },
        "scientific_gate": {
            "all_required_cache_jobs_verified": True,
            "cache_delivery_publicly_frozen_before_staging": True,
            "scientific_endpoints_scored": False,
            "scientific_endpoints_printed": False,
            "cache_npz_payloads_opened_or_inspected_by_stager": False,
            "one_shot_scoring_permitted": True,
        },
        "stager": {
            "file": Path(__file__).resolve().name,
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    index_path = output_root / "score_input_index.json"
    index_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-job-plan", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--public-plan-commit", required=True)
    parser.add_argument("--public-delivery-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"scorer staging root must start absent: {args.out}")
    plan, delivery, cache_job_plan = validate_plan_delivery(
        args.plan,
        args.cache_job_plan,
        args.delivery,
        args.public_plan_commit,
        args.public_delivery_commit,
    )
    payload = stage_mode(
        mode=args.mode,
        input_root=args.input_root,
        output_root=args.out,
        plan_path=args.plan,
        cache_job_plan_path=args.cache_job_plan,
        plan=plan,
        cache_job_plan=cache_job_plan,
        delivery=delivery,
        public_plan_commit=args.public_plan_commit,
        public_delivery_commit=args.public_delivery_commit,
    )
    print("staged held-out jobs:", payload["counts"]["jobs"])
    print("staged cache manifests:", payload["counts"]["cache_manifests"])
    print("staged sealed cache files:", payload["counts"]["sealed_cache_files"])
    print("score-input payload SHA-256:", payload["payload_sha256"])
    print("PUBLIC_HELDOUT_DELIVERY_VERIFIED_AND_STAGED_WITHOUT_NPZ_ACCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
