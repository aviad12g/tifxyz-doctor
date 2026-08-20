#!/usr/bin/env python3
"""Upload the frozen bundle and start the sealed RunPod development executor."""

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
    raise RuntimeError("RunPod SSH endpoint did not become ready")


def run_checked(command: list[str], timeout: int) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
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
    manifest = load_hashed(args.bundle / "bundle_manifest.json")
    receipt = json.loads(args.provider_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") not in {
        "POD_RESUMED_AWAITING_DEPLOYMENT",
        "PODS_ALLOCATED_AWAITING_DEPLOYMENT",
    }:
        raise RuntimeError("provider receipt is not ready for first deployment")
    if receipt.get("selected_hardware") is not None and receipt.get("private_bundle_egress_permitted") is not True:
        raise RuntimeError("substitute provider receipt lacks explicit private-bundle egress approval")
    if receipt.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("provider receipt is bound to another plan")
    if manifest.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("bundle is bound to another plan")
    job_ids = [job["job_id"] for job in plan["jobs"]]
    if manifest.get("job_ids") != job_ids:
        raise RuntimeError("bundle job order mismatch")
    import runpod

    pods = receipt.get("pods")
    if pods is None:
        pod = dict(receipt["pod"])
        pod["jobs"] = job_ids
        pods = [pod]
    if [job for pod in pods for job in pod["jobs"]] != job_ids:
        raise RuntimeError("provider pod partitions do not reconstruct the frozen job order")
    deployments = []
    try:
        for pod_index, pod in enumerate(pods):
            host, port = ssh_target(runpod, pod["id"])
            remote_root = (
                f"/workspace/gapbalance-development-{receipt['public_runpod_commit'][:8]}-p{pod_index}"
            )
            ssh = [
                "ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15", f"root@{host}",
            ]
            run_checked(
                [*ssh, f"test ! -e {shlex.quote(remote_root)} && mkdir -p {shlex.quote(remote_root + '/bundle')}"],
                60,
            )
            rsync_ssh = f"ssh -p {port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
            run_checked(
                [
                    "rsync", "--archive", "--copy-links", "--partial", "--protect-args",
                    "--info=progress2", "-e", rsync_ssh, str(args.bundle) + "/",
                    f"root@{host}:{remote_root}/bundle/",
                ],
                2 * 60 * 60,
            )
            remote = [
                "python", f"{remote_root}/bundle/controller/execute_gapbalance_runpod_development.py",
                "--plan", f"{remote_root}/bundle/controller/GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json",
                "--bundle-root", f"{remote_root}/bundle",
                "--bundle-manifest", f"{remote_root}/bundle/bundle_manifest.json",
                "--input", f"{remote_root}/bundle/input",
                "--launchers", f"{remote_root}/bundle/launchers",
                "--wrapper", f"{remote_root}/bundle/controller/run_gapbalance_runpod_development_job.py",
                "--working", f"{remote_root}/output",
                "--temp", f"{remote_root}/temp",
                "--status", f"{remote_root}/status/status.json",
            ]
            if len(pods) > 1:
                remote.extend([
                    "--allocation-layout",
                    f"{remote_root}/bundle/controller/GAPBALANCE_RUNPOD_MULTI_POD_RETRY.json",
                ])
                for job_id in pod["jobs"]:
                    remote.extend(["--job-id", job_id])
            command = (
                "mkdir -p " + shlex.quote(remote_root + "/status") + " && nohup "
                + " ".join(shlex.quote(part) for part in remote)
                + f" > {shlex.quote(remote_root + '/status/controller.operational.log')} 2>&1 < /dev/null & echo $!"
            )
            remote_pid = run_checked([*ssh, command], 60)
            if not remote_pid.isdigit():
                raise RuntimeError(f"remote executor did not return PID: {remote_pid!r}")
            deployments.append({
                "pod_id": pod["id"],
                "jobs": pod["jobs"],
                "remote_root": remote_root,
                "remote_executor_pid": int(remote_pid),
                "ssh_host": host,
                "ssh_port": port,
            })
    except Exception:
        for pod in pods:
            try:
                runpod.stop_pod(pod["id"])
            except Exception:
                pass
        receipt["status"] = "DEPLOYMENT_ERROR_PODS_STOPPED"
        receipt["deployment_error_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.provider_receipt, receipt)
        raise
    receipt["deployment"] = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_payload_sha256": manifest["payload_sha256"],
        "pods": deployments,
    }
    receipt["status"] = "RUNPOD_DEVELOPMENT_EXECUTOR_RUNNING_NOT_SCORED"
    atomic_json(args.provider_receipt, receipt)
    print(json.dumps(receipt["deployment"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
