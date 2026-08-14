#!/usr/bin/env python3
"""Resume the frozen CPU pod and start the private-Kaggle transport pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


POD_ID = "z0vqb3dt43vkga"


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
    if not isinstance(payload, dict) or payload.get("payload_sha256") != canonical_sha256(payload):
        raise RuntimeError(f"invalid hashed JSON: {path}")
    return payload


def write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def tree_identity(root: Path) -> dict:
    records = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"transfer source tree contains symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        records.append(f"{sha256_file(path)}  {size}  {path.relative_to(root).as_posix()}\n")
        total += size
    return {
        "files": len(records),
        "bytes": total,
        "ledger_sha256": hashlib.sha256("".join(records).encode()).hexdigest(),
    }


def run(command: list[str], *, timeout: int) -> None:
    completed = subprocess.run(command, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command[:4]}")


def ssh_endpoint(runpod) -> tuple[str, int]:
    for _ in range(90):
        pod = runpod.get_pod(POD_ID)
        runtime = pod.get("runtime")
        if isinstance(runtime, dict):
            for port in runtime.get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    return port["ip"], int(port["publicPort"])
        time.sleep(2)
    raise RuntimeError("RunPod SSH endpoint did not become ready")


def validate_pod(pod: dict) -> None:
    if (
        pod.get("desiredStatus") != "RUNNING"
        or int(pod.get("gpuCount") or 0) != 0
        or int(pod.get("vcpuCount") or 0) != 32
        or int(pod.get("memoryInGb") or 0) != 128
        or float(pod.get("costPerHr") or 0.0) > 1.28
    ):
        raise RuntimeError("resumed RunPod allocation is outside the frozen CPU-only shape")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--scoring-assets", type=Path, required=True)
    parser.add_argument("--frozen-thresholds", type=Path, required=True)
    parser.add_argument("--metric-archive", type=Path, required=True)
    parser.add_argument("--metric-runtime", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--metric-verifier", type=Path, required=True)
    parser.add_argument("--puller", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--runtime-preparer", type=Path, required=True)
    parser.add_argument("--runtime-bootstrap-plan", type=Path, required=True)
    parser.add_argument("--runtime-predecessor-plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.parent.exists():
        raise RuntimeError("retry receipt target must start absent")
    plan = load_hashed(args.plan)
    prior = load_hashed(args.prior_receipt)
    if prior.get("payload_sha256") != plan["private_kaggle_transport_retry"]["prior_receipt"]["payload_sha256"]:
        raise RuntimeError("prior stopped receipt differs from frozen plan")
    if (
        not args.credentials.is_file()
        or args.credentials.is_symlink()
        or stat.S_IMODE(args.credentials.stat().st_mode) & 0o077
    ):
        raise RuntimeError("local Kaggle credentials are absent or too permissive")
    exact_files = {
        args.launcher: plan["embedded_real_launcher"],
        args.executor: plan["remote_executor"],
        args.metric_verifier: plan["public_metric_verifier"],
        args.puller: plan["private_transport_puller"],
        args.wrapper: plan["private_transport_wrapper"],
        args.runtime_preparer: plan["runtime_preparer"],
        args.runtime_bootstrap_plan: plan["runtime_bootstrap_plan"],
        args.runtime_predecessor_plan: plan["runtime_predecessor_plan"],
        args.metric_archive: plan["public_metric_archive"],
        args.frozen_thresholds: plan["frozen_thresholds"],
    }
    for path, record in exact_files.items():
        if identity(path) != record:
            raise RuntimeError(f"local public artifact differs from frozen plan: {path.name}")
    for root, record in (
        (args.wheelhouse, plan["private_kaggle_transport"]["kagglehub_wheelhouse_tree"]),
        (args.scoring_assets, plan["scoring_assets_tree"]),
        (args.metric_runtime, plan["metric_runtime_tree"]),
    ):
        if tree_identity(root) != record:
            raise RuntimeError(f"local public tree differs from frozen plan: {root.name}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PRIVATE_KAGGLE_CPU_RETRY_PREFLIGHT_PASSED",
                    "plan_payload_sha256": plan["payload_sha256"],
                    "pod_id": POD_ID,
                    "gpu_count": 0,
                    "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = {
        "schema_version": "1.0",
        "status": "CPU-only private-Kaggle retry intent recorded before provider mutation",
        "pod_id": POD_ID,
        "plan_payload_sha256": plan["payload_sha256"],
        "plan_file_sha256": sha256_file(args.plan),
        "public_runpod_commit": plan["public_runpod_commit"],
        "compute_type": "CPU",
        "gpu_count": 0,
        "vcpu_count": 32,
        "memory_gb": 128,
        "price_usd_per_hour": 1.28,
        "absolute_campaign_cap_usd": 13.5,
        "prior_guarded_spend_upper_bound_usd": plan["budget"]["prior_guarded_spend_upper_bound_usd"],
        "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
        "guard_seconds": plan["budget"]["guard_seconds"],
        "scientific_outputs_inspected": False,
    }
    args.receipt.parent.mkdir(parents=True)
    write_hashed(args.receipt, receipt)
    import runpod

    try:
        runpod.resume_pod(POD_ID, 0)
        pod = None
        for _ in range(90):
            pod = runpod.get_pod(POD_ID)
            if pod.get("desiredStatus") == "RUNNING":
                break
            time.sleep(2)
        if not pod:
            raise RuntimeError("RunPod did not return the resumed CPU pod")
        validate_pod(pod)
        receipt["created_at"] = datetime.now(timezone.utc).isoformat()
        receipt["status"] = "CPU-only RunPod retry resumed within frozen campaign cap"
        write_hashed(args.receipt, receipt)
        host, port = ssh_endpoint(runpod)
        receipt["ssh_host"] = host
        receipt["ssh_port"] = port
        write_hashed(args.receipt, receipt)
        ssh = [
            "ssh", "-p", str(port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            f"root@{host}",
        ]
        absent = [
            "/workspace/real-scoring-controller-v11",
            "/workspace/real-scoring-pipeline-status-v11",
            "/workspace/real-transport-status-v11",
            "/workspace/real-transport-download-v1",
            "/workspace/real-scoring-input-kaggle-v1",
            "/workspace/real-scoring-status-v11",
            "/workspace/real-scoring-work-v11",
            "/workspace/kagglehub-runtime-v1",
            "/workspace/kagglehub-wheelhouse-v1",
            "/workspace/kaggle-credential-v1",
            "/workspace/real-scoring-public",
            "/workspace/bundle",
        ]
        run(ssh + [" && ".join(f"test ! -e {path}" for path in absent)], timeout=30)
        run(
            ssh + [
                "mkdir -p /workspace/real-scoring-controller-v11 "
                "/workspace/kagglehub-wheelhouse-v1 /workspace/kaggle-credential-v1 "
                "/workspace/real-scoring-public/metric-runtime "
                "/workspace/bundle/input/assets /workspace/bundle/input/threshold-freeze"
            ],
            timeout=30,
        )
        rsync = [
            "rsync", "--archive", "--copy-links", "--partial", "--protect-args",
            "-e", f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15",
        ]
        run(rsync + [str(args.wheelhouse) + "/", f"root@{host}:/workspace/kagglehub-wheelhouse-v1/"], timeout=300)
        run(rsync + [str(args.scoring_assets) + "/", f"root@{host}:/workspace/bundle/input/assets/"], timeout=300)
        run(rsync + [str(args.metric_runtime) + "/", f"root@{host}:/workspace/real-scoring-public/metric-runtime/"], timeout=900)
        controller_files = [
            args.plan, args.launcher, args.executor, args.metric_verifier, args.puller,
            args.wrapper, args.runtime_preparer, args.runtime_bootstrap_plan,
            args.runtime_predecessor_plan, args.metric_archive,
        ]
        run(rsync + [*map(str, controller_files), f"root@{host}:/workspace/real-scoring-controller-v11/"], timeout=900)
        run(rsync + [str(args.frozen_thresholds), f"root@{host}:/workspace/bundle/input/threshold-freeze/"], timeout=300)
        run(
            [
                "scp", "-P", str(port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
                str(args.credentials), f"root@{host}:/workspace/kaggle-credential-v1/kaggle.json",
            ],
            timeout=120,
        )
        run(ssh + ["chmod 600 /workspace/kaggle-credential-v1/kaggle.json"], timeout=30)
        remote = (
            "nohup python3 /workspace/real-scoring-controller-v11/runpod_prepare_and_execute_from_kaggle.py "
            "--plan /workspace/real-scoring-controller-v11/runpod_real_scoring_private_kaggle_transport_plan.json "
            "--runtime-predecessor-plan /workspace/real-scoring-controller-v11/runpod_real_scoring_plan.json "
            "--runtime-bootstrap-plan /workspace/real-scoring-controller-v11/runpod_real_scoring_runtime_bootstrap_plan.json "
            "--runtime-preparer /workspace/real-scoring-controller-v11/prepare_runpod_real_scoring_runtime.py "
            "--puller /workspace/real-scoring-controller-v11/pull_runpod_real_transport_dataset.py "
            "--executor /workspace/real-scoring-controller-v11/runpod_execute_real_scoring.py "
            "--launcher /workspace/real-scoring-controller-v11/one_shot_scoring_launcher.py "
            "--metric-verifier /workspace/real-scoring-controller-v11/verify_official_metric.py "
            "--metric-archive /workspace/real-scoring-controller-v11/vesuvius-metric-resources.zip "
            "--wheelhouse /workspace/kagglehub-wheelhouse-v1 "
            "--credentials /workspace/kaggle-credential-v1/kaggle.json "
            "--pipeline-status-root /workspace/real-scoring-pipeline-status-v11 "
            "--transport-status-root /workspace/real-transport-status-v11 "
            "--download-root /workspace/real-transport-download-v1 "
            "--input-root /workspace/real-scoring-input-kaggle-v1 "
            "--scoring-status-root /workspace/real-scoring-status-v11 "
            "--working-root /workspace/real-scoring-work-v11 "
            ">/workspace/real-scoring-bootstrap-v11.log 2>&1 </dev/null &"
        )
        run(ssh + [remote], timeout=30)
        receipt["status"] = "CPU-only private-Kaggle transport and exact scorer pipeline started"
        receipt["executor_started_at"] = datetime.now(timezone.utc).isoformat()
        write_hashed(args.receipt, receipt)
        print("RUNPOD_PRIVATE_KAGGLE_REAL_SCORING_ACCEPTED")
        return 0
    except Exception as error:
        try:
            runpod.stop_pod(POD_ID)
        finally:
            receipt["status"] = "CPU-only private-Kaggle retry failed operationally and billing was stopped"
            receipt["error_type"] = type(error).__name__
            receipt["error_message"] = str(error)[:500]
            receipt["stopped_at"] = datetime.now(timezone.utc).isoformat()
            write_hashed(args.receipt, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
