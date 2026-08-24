#!/usr/bin/env python3
"""Transfer the frozen bundle and start the sealed RunPod primary executor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def ssh_target(runpod, pod_id: str) -> tuple[str, int]:
    for _ in range(120):
        pod = runpod.get_pod(pod_id)
        runtime = (pod or {}).get("runtime") or {}
        for port in runtime.get("ports") or []:
            if (
                int(port.get("privatePort") or -1) == 22
                and port.get("isIpPublic")
                and port.get("ip")
                and port.get("publicPort")
            ):
                return str(port["ip"]), int(port["publicPort"])
        time.sleep(5)
    raise RuntimeError(f"public SSH endpoint did not become ready: {pod_id}")


def run_checked(command: list[str], timeout: int) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        message = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"command failed ({result.returncode}): {message[-4000:]}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    plan = load_hashed(args.plan)
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    manifest = load_hashed(args.bundle / "bundle_manifest.json")
    if receipt.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("provider receipt is bound to another plan")
    if manifest.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("bundle is bound to another plan")
    jobs = [item["job_id"] for item in plan["jobs"]]
    if manifest.get("job_ids") != jobs:
        raise RuntimeError("deployment bundle is not the exact seven-job primary")
    pods = receipt.get("pods")
    if not isinstance(pods, list) or len(pods) != 1 or pods[0].get("jobs") != jobs:
        raise RuntimeError(
            "this frozen deployer requires the preferred one-pod seven/eight-GPU layout"
        )
    import runpod

    pod_id = pods[0]["id"]
    host, port = ssh_target(runpod, pod_id)
    ssh = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        f"root@{host}",
    ]
    run_checked([*ssh, "mkdir -p /workspace/bundle /workspace/output /workspace/temp /workspace/primary-status"], 60)
    rsync_ssh = (
        f"ssh -p {port} -o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=15"
    )
    run_checked(
        [
            "rsync",
            "--archive",
            "--copy-links",
            "--partial",
            "--protect-args",
            "--info=progress2",
            "-e",
            rsync_ssh,
            str(args.bundle) + "/",
            f"root@{host}:/workspace/bundle/",
        ],
        2 * 60 * 60,
    )
    remote = [
        "python",
        "/workspace/bundle/controller/runpod_execute_primary.py",
        "--plan",
        "/workspace/bundle/controller/runpod_synthetic_replacement_plan.json",
        "--bundle-root",
        "/workspace/bundle",
        "--bundle-manifest",
        "/workspace/bundle/bundle_manifest.json",
        "--input",
        "/workspace/bundle/input",
        "--launchers",
        "/workspace/bundle/launchers",
        "--wrapper",
        "/workspace/bundle/controller/runpod_launch_frozen_job.py",
        "--working",
        "/workspace/output",
        "--temp",
        "/workspace/temp",
        "--status",
        "/workspace/primary-status/status.json",
    ]
    command = (
        "nohup "
        + " ".join(shlex.quote(part) for part in remote)
        + " > /workspace/primary-status/controller.operational.log 2>&1 < /dev/null & echo $!"
    )
    remote_pid = run_checked([*ssh, command], 60)
    if not remote_pid.isdigit():
        raise RuntimeError(f"remote executor did not return a PID: {remote_pid!r}")
    receipt["deployment"] = {
        "bundle_manifest_payload_sha256": manifest["payload_sha256"],
        "deployed_at": now(),
        "remote_executor_pid": int(remote_pid),
        "ssh_host": host,
        "ssh_port": port,
    }
    receipt["status"] = "PRIMARY_EXECUTOR_RUNNING_SEALED"
    receipt["scientific_outputs_inspected"] = False
    atomic_json(args.provider_receipt, receipt)
    print(json.dumps(receipt["deployment"], indent=2, sort_keys=True))
    print("RUNPOD_PRIMARY_EXECUTOR_STARTED_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
