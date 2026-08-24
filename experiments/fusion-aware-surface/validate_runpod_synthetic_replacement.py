#!/usr/bin/env python3
"""Fail-closed validation for the frozen RunPod synthetic replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def validate_payload(payload: dict, label: str) -> None:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} has no canonical payload SHA-256")
    body = dict(payload)
    body.pop("payload_sha256")
    observed = sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed != expected:
        raise RuntimeError(f"{label} payload SHA-256 mismatch: {observed}")


def validate_asset_ledger(root: Path) -> dict:
    ledger = root / "SOURCE_SHA256SUMS"
    if not ledger.is_file():
        raise RuntimeError("frozen asset ledger is absent")
    if sha256_file(ledger) != "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba":
        raise RuntimeError("frozen asset ledger identity mismatch")
    records = []
    total = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"unsafe or missing asset: {relative}")
        records.append(relative)
        total += path.stat().st_size
    if len(records) != 278:
        raise RuntimeError(f"asset ledger record count mismatch: {len(records)}")
    source_checkpoint = root / "model" / "Model_epoch499.pth"
    if (
        not source_checkpoint.is_file()
        or source_checkpoint.stat().st_size != 819_171_665
        or sha256_file(source_checkpoint)
        != "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
    ):
        raise RuntimeError("source checkpoint identity mismatch")
    return {
        "ledger_sha256": "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba",
        "record_count": len(records),
        "ledger_payload_bytes": total,
        "remote_full_hash_verification_required_before_inference": True,
        "source_checkpoint_bytes": source_checkpoint.stat().st_size,
        "source_checkpoint_sha256": "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f",
    }


def find_exact_file(root: Path, suffix: str, digest: str) -> Path:
    matches = [
        path
        for path in root.rglob(suffix)
        if path.is_file() and sha256_file(path) == digest
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix} with SHA-256 {digest}; found {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--replacement-plan", type=Path, default=here / "runpod_synthetic_replacement_plan.json")
    parser.add_argument("--heldout-plan", type=Path, default=here / "heldout_execution_plan.json")
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=here / "frozen_thresholds.json")
    parser.add_argument("--threshold-manifest", type=Path, default=here / "threshold_run_manifest.json")
    args = parser.parse_args()

    replacement = load_json(args.replacement_plan)
    heldout = load_json(args.heldout_plan)
    index = load_json(args.packages / "generated_packages_index.json")
    for payload, label in (
        (replacement, "replacement plan"),
        (heldout, "held-out plan"),
        (index, "package index"),
    ):
        validate_payload(payload, label)

    provenance = replacement["provenance"]
    if heldout["payload_sha256"] != provenance["heldout_execution_plan_payload_sha256"]:
        raise RuntimeError("replacement does not bind this held-out plan")
    if index["payload_sha256"] != provenance["generated_packages_index_payload_sha256"]:
        raise RuntimeError("replacement does not bind this generated package index")
    if replacement["scientific_contract"] != {
        "heldout_outputs_inspected_before_replacement_freeze": False,
        "kaggle_result_may_override_runpod_result": False,
        "models_thresholds_seeds_cells_shards_endpoints_or_gates_changed": False,
        "partial_run_scoring_permitted": False,
        "platform_disagreement_policy": "report both faithfully, investigate only operational causes, and never select the more favorable outcome",
        "reason_for_replacement": "reduce execution latency only",
        "result_inspection_before_all_primary_jobs_complete_and_delivery_is_frozen": False,
        "runpod_result_is_primary_even_if_adverse": True,
        "test_time_tuning_permitted": False,
    }:
        raise RuntimeError("scientific replacement contract mismatch")
    if replacement["budget"]["hard_total_cap_usd"] != 20.0:
        raise RuntimeError("hard budget cap is not exactly USD 20")
    if replacement["budget"]["billing_cutoff_usd"] > 18.5:
        raise RuntimeError("billing cutoff exceeds frozen reserve policy")
    for identity in replacement["tooling"].values():
        path = args.replacement_plan.parent / identity["file"]
        if (
            not path.is_file()
            or path.stat().st_size != identity["bytes"]
            or sha256_file(path) != identity["sha256"]
        ):
            raise RuntimeError(f"replacement tooling identity mismatch: {identity['file']}")

    synthetic_index = {
        item["job_id"]: item
        for item in index["packages"]
        if item["mode"] == "synthetic_ray_cache"
    }
    scientific_jobs = {item["job_id"]: item for item in heldout["synthetic_ray_jobs"]}
    checkpoints_by_digest: dict[str, list[Path]] = {}
    for checkpoint in args.checkpoints.rglob("*.pth"):
        if checkpoint.is_file():
            checkpoints_by_digest.setdefault(sha256_file(checkpoint), []).append(checkpoint)
    if list(item["job_id"] for item in replacement["jobs"]) != [
        item["job_id"] for item in heldout["synthetic_ray_jobs"]
    ]:
        raise RuntimeError("seven-job order changed")
    if len(replacement["jobs"]) != 7:
        raise RuntimeError("replacement must contain exactly seven jobs")

    for job in replacement["jobs"]:
        job_id = job["job_id"]
        package = synthetic_index.get(job_id)
        original = scientific_jobs.get(job_id)
        if package is None or original is None:
            raise RuntimeError(f"unknown job: {job_id}")
        launcher = args.packages / package["directory"] / "heldout_cache_launcher.py"
        if (
            launcher.stat().st_size != job["launcher_bytes"]
            or sha256_file(launcher) != job["launcher_sha256"]
            or package["files"]["heldout_cache_launcher.py"]
            != {"bytes": job["launcher_bytes"], "sha256": job["launcher_sha256"]}
        ):
            raise RuntimeError(f"launcher identity mismatch: {job_id}")
        for key in ("run", "model_state_sha256", "selected_threshold", "cell_count", "shard_indices"):
            if job.get(key) != original.get(key):
                raise RuntimeError(f"scientific field changed for {job_id}: {key}")
        if job["shard_indices"] != list(range(10)):
            raise RuntimeError(f"shard membership mismatch: {job_id}")
        if job["model_state_sha256"] is not None:
            matches = checkpoints_by_digest.get(job["model_state_sha256"], [])
            if len(matches) != 1:
                raise RuntimeError(f"checkpoint identity ambiguity: {job_id}: {matches}")
            checkpoint = matches[0]
            mount = checkpoint.parents[3]
            manifests = list(mount.rglob("training_run_manifest.json"))
            if len(manifests) != 1:
                raise RuntimeError(f"training manifest ambiguity: {job_id}")
            if sha256_file(manifests[0]) != original["training_run_manifest_sha256"]:
                raise RuntimeError(f"training manifest mismatch: {job_id}")

    threshold_binding = heldout["threshold_binding"]
    if sha256_file(args.thresholds) != threshold_binding["frozen_thresholds"]["sha256"]:
        raise RuntimeError("frozen thresholds file mismatch")
    if sha256_file(args.threshold_manifest) != threshold_binding["threshold_run_manifest"]["sha256"]:
        raise RuntimeError("threshold manifest mismatch")
    assets = validate_asset_ledger(args.assets)
    receipt = {
        "assets": assets,
        "job_count": 7,
        "package_index_payload_sha256": index["payload_sha256"],
        "replacement_plan_payload_sha256": replacement["payload_sha256"],
        "status": "RUNPOD_REPLACEMENT_INPUTS_VALID",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
