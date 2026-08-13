#!/usr/bin/env python3
"""Upload frozen real-scoring inputs to the resumed RunPod allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


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
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ssh_target(runpod, pod_id: str) -> tuple[str, int]:
    for _ in range(90):
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime")
        if isinstance(runtime, dict):
            for port in runtime.get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    return port["ip"], int(port["publicPort"])
        time.sleep(2)
    raise RuntimeError("RunPod SSH endpoint did not become ready")


def run(command: list[str], *, timeout: int) -> None:
    completed = subprocess.run(command, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command[:4]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sealed-input-root", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    parser.add_argument("--metric-runtime", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    args = parser.parse_args()
    plan = load_hashed(args.plan)
    receipt = load_hashed(args.receipt)
    if receipt.get("status") != "RunPod allocation resumed for sealed real scoring":
        raise RuntimeError("RunPod allocation is not in the expected resumed state")
    if receipt["plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("receipt points to another RunPod plan")
    import runpod

    host, port = ssh_target(runpod, receipt["pod_id"])
    ssh = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        f"root@{host}",
    ]
    run(ssh + ["test ! -e /workspace/real-scoring-status && test ! -e /workspace/real-scoring-input"], timeout=30)
    run(ssh + ["mkdir -p /workspace/real-scoring-controller /workspace/real-scoring-public"], timeout=30)
    receipt["status"] = "authorized sealed real-cache upload in progress"
    receipt["deployment_started_at"] = datetime.now(timezone.utc).isoformat()
    receipt["ssh_host"] = host
    receipt["ssh_port"] = port
    write_hashed(args.receipt, receipt)
    rsync_base = [
        "rsync",
        "--archive",
        "--copy-links",
        "--partial",
        "--protect-args",
        "-e",
        f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15",
    ]
    run(rsync_base + [str(args.sealed_input_root) + "/", f"root@{host}:/workspace/real-scoring-input/"], timeout=3600)
    run(rsync_base + [str(args.metric_source) + "/", f"root@{host}:/workspace/real-scoring-public/metric-source/"], timeout=900)
    run(rsync_base + [str(args.metric_runtime) + "/", f"root@{host}:/workspace/real-scoring-public/metric-runtime/"], timeout=900)
    run(
        rsync_base
        + [
            str(args.plan),
            str(args.launcher),
            str(args.executor),
            f"root@{host}:/workspace/real-scoring-controller/",
        ],
        timeout=300,
    )
    remote = (
        "nohup python3 /workspace/real-scoring-controller/runpod_execute_real_scoring.py "
        "--plan /workspace/real-scoring-controller/runpod_real_scoring_plan.json "
        "--input-manifest /workspace/real-scoring-input/runpod_real_input_manifest.json "
        "--input-root /workspace/real-scoring-input "
        "--launcher /workspace/real-scoring-controller/one_shot_scoring_launcher.py "
        "--status-root /workspace/real-scoring-status "
        "--working-root /workspace/real-scoring-work "
        ">/workspace/real-scoring-bootstrap.log 2>&1 </dev/null &"
    )
    run(ssh + [remote], timeout=30)
    receipt["status"] = "sealed real-scoring executor started"
    receipt["executor_started_at"] = datetime.now(timezone.utc).isoformat()
    write_hashed(args.receipt, receipt)
    print("RUNPOD_REAL_SCORING_EXECUTOR_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
