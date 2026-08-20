#!/usr/bin/env python3
"""Copy and mechanically verify all 12 RunPod caches without opening NPZ."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


PREFIX = b"# GAPBALANCE_DEVELOPMENT_JOB_CONFIG_HEX="


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def embedded_config(path: Path) -> dict:
    first = path.read_bytes().split(b"\n", 1)[0]
    if not first.startswith(PREFIX):
        raise RuntimeError(f"launcher config absent: {path}")
    payload = json.loads(bytes.fromhex(first[len(PREFIX):].decode("ascii")))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError(f"launcher config hash mismatch: {path}")
    return payload


def run_checked(command: list[str], timeout: int) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        message = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"command failed ({result.returncode}): {message[-4000:]}")
    return result.stdout.strip()


def verify_job(root: Path, job: dict, config: dict) -> dict:
    output = root / job["job_id"] / "gapbalance-development"
    if not output.is_dir():
        raise RuntimeError(f"output root absent: {job['job_id']}")
    index_path = output / "gapbalance_development_job_index.json"
    index = load_hashed(index_path)
    expected_runs = {run: identity["checkpoint_sha256"] for run, identity in config["runs"].items()}
    if (
        index.get("status") != "GapBalance development cache job complete without endpoint scoring"
        or index.get("job_id") != job["job_id"]
        or index.get("mode") != "synthetic"
        or index.get("seed") != job["seed"]
        or index.get("shard_index") != job["shard_index"]
        or index.get("runs") != expected_runs
        or index.get("selected_thresholds") is not False
        or index.get("scientific_endpoints_scored") is not False
        or index.get("confirmation_outputs_inspected") is not False
    ):
        raise RuntimeError(f"job index identity mismatch: {job['job_id']}")
    records = index.get("files")
    if not isinstance(records, list):
        raise RuntimeError(f"job index file records absent: {job['job_id']}")
    expected_files = {record["file"] for record in records} | {index_path.name}
    observed_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise RuntimeError(f"output file-set mismatch: {job['job_id']}")
    for record in records:
        path = output / record["file"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"output identity mismatch: {job['job_id']}:{record['file']}")
    npz_records = [record for record in records if record["file"].endswith(".npz")]
    manifest_records = [record for record in records if record["file"].endswith(".json")]
    if len(npz_records) != 60 or len(manifest_records) != 3:
        raise RuntimeError(f"output count mismatch: {job['job_id']}")
    for record in manifest_records:
        manifest = load_hashed(output / record["file"])
        run = manifest.get("run")
        if (
            run not in config["runs"]
            or manifest.get("model_state_sha256") != config["runs"][run]["checkpoint_sha256"]
            or manifest.get("selected_threshold") is not None
            or manifest.get("confirmation_outputs_inspected") is not False
        ):
            raise RuntimeError(f"cache manifest mismatch: {job['job_id']}:{record['file']}")
    return {
        "job_id": job["job_id"],
        "job_index": str(index_path),
        "job_index_sha256": sha256_file(index_path),
        "job_index_payload_sha256": index["payload_sha256"],
        "npz_files_hashed_not_opened": 60,
        "cache_manifests_verified": 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("local RunPod collection output must start absent")
    plan = load_hashed(args.plan)
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "RUNPOD_DEVELOPMENT_EXECUTOR_RUNNING_NOT_SCORED":
        raise RuntimeError("provider receipt is not in the expected running state")
    if receipt.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("provider receipt is bound to another plan")
    job_ids = [job["job_id"] for job in plan["jobs"]]
    deployments = receipt["deployment"].get("pods")
    if deployments is None:
        deployment = dict(receipt["deployment"])
        deployment["pod_id"] = receipt["pod"]["id"]
        deployment["jobs"] = job_ids
        deployments = [deployment]
    if [job for deployment in deployments for job in deployment["jobs"]] != job_ids:
        raise RuntimeError("deployment partitions do not reconstruct the frozen job order")
    args.out.mkdir(parents=True)
    outputs = args.out / "outputs"
    operational = args.out / "operational"
    outputs.mkdir()
    operational.mkdir()
    for deployment in deployments:
        host, port = deployment["ssh_host"], int(deployment["ssh_port"])
        remote_root = deployment["remote_root"]
        ssh = [
            "ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=15", f"root@{host}",
        ]
        status_raw = run_checked([*ssh, f"cat {remote_root}/status/status.json"], 60)
        status = json.loads(status_raw)
        if (
            status.get("state") != "COMPLETE"
            or status.get("plan_payload_sha256") != plan["payload_sha256"]
            or status.get("scientific_endpoints_scored") is not False
            or status.get("confirmation_outputs_inspected") is not False
            or list(status.get("jobs", {})) != deployment["jobs"]
        ):
            raise RuntimeError(f"remote RunPod status is not sealed COMPLETE: {deployment['pod_id']}")
        for job_id in deployment["jobs"]:
            record = status["jobs"][job_id]
            if record.get("state") != "COMPLETE" or record.get("returncode") != 0:
                raise RuntimeError(f"remote job is not complete: {job_id}")
        rsync_ssh = f"ssh -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
        run_checked(
            ["rsync", "--archive", "--partial", "--protect-args", "-e", rsync_ssh,
             f"root@{host}:{remote_root}/output/", str(outputs) + "/"],
            2 * 60 * 60,
        )
        pod_operational = operational / deployment["pod_id"]
        pod_operational.mkdir()
        run_checked(
            ["rsync", "--archive", "--partial", "--protect-args", "-e", rsync_ssh,
             f"root@{host}:{remote_root}/status/", str(pod_operational) + "/"],
            30 * 60,
        )
    observed_top = {path.name for path in outputs.iterdir() if path.is_dir()}
    if observed_top != set(job_ids) or any(path.is_file() for path in outputs.iterdir()):
        raise RuntimeError("copied RunPod top-level output set mismatch")
    verified = []
    for job in plan["jobs"]:
        launcher = args.packages / job["job_id"] / "gapbalance_development_kaggle_launcher.py"
        config = embedded_config(launcher)
        if config.get("payload_sha256") != job["config_payload_sha256"]:
            raise RuntimeError(f"local package config mismatch: {job['job_id']}")
        verified.append(verify_job(outputs, job, config))
        print(f"RUNPOD_DEVELOPMENT_JOB_VERIFIED_NOT_OPENED job={job['job_id']}", flush=True)
    import runpod

    frozen_pods = receipt.get("pods") or [receipt["pod"]]
    for pod in frozen_pods:
        runpod.stop_pod(pod["id"])
    stopped_at = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(receipt["billing_started_at"])
    rate = sum(float(pod["aggregate_hourly_rate_usd"]) for pod in frozen_pods)
    spend = rate * max(
        0.0, (stopped_at - started_at).total_seconds()
    ) / 3600.0
    if spend > float(receipt["development_billing_cutoff_usd"]):
        raise RuntimeError("conservative development spend exceeded the public cutoff")
    collection = {
        "schema_version": "1.0",
        "status": "12 authoritative RunPod development caches verified without opening NPZ",
        "plan_payload_sha256": plan["payload_sha256"],
        "provider_pod_ids": [pod["id"] for pod in frozen_pods],
        "provider_stopped_at": stopped_at.isoformat(),
        "conservative_development_spend_usd": round(spend, 6),
        "jobs": verified,
        "counts": {"jobs": 12, "npz_files_hashed_not_opened": 720, "cache_manifests_verified": 36},
        "scientific_endpoints_scored": False,
        "npz_payloads_opened": False,
        "confirmation_outputs_inspected": False,
    }
    collection["payload_sha256"] = canonical(collection)
    (args.out / "runpod_development_collection.json").write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt.update(
        status="RUNPOD_DEVELOPMENT_STOPPED_AFTER_RESULT_BLIND_COLLECTION",
        provider_stopped_at=stopped_at.isoformat(),
        conservative_development_spend_usd=round(spend, 6),
        collection_payload_sha256=collection["payload_sha256"],
    )
    temp = args.provider_receipt.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.provider_receipt)
    print("ALL_12_RUNPOD_DEVELOPMENT_CACHES_VERIFIED_NOT_SCORED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
