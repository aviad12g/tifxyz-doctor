#!/usr/bin/env python3
"""Create, monitor, and terminate the frozen CPU-only RunPod real scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    if path.exists() or path.parent.exists():
        raise RuntimeError("receipt target must start absent")
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content.pop("payload_sha256", None)
    content["payload_sha256"] = canonical_sha256(content)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def spend(receipt: dict, *, guard_seconds: int = 0) -> float:
    elapsed = max(
        0.0,
        (datetime.now(timezone.utc) - parse_time(receipt["created_at"])).total_seconds(),
    )
    return receipt["price_usd_per_hour"] * (elapsed + guard_seconds) / 3600.0


def load_runpod():
    import runpod

    return runpod


def validate_running_pod(pod: dict, plan: dict) -> None:
    provider = plan["provider"]
    price = float(pod.get("costPerHr") or 0.0)
    if (
        int(pod.get("gpuCount") or 0) != 0
        or int(pod.get("vcpuCount") or 0) < provider["vcpu_count"]
        or int(pod.get("memoryInGb") or 0) < provider["minimum_memory_gb"]
        or price <= 0.0
        or price > provider["maximum_price_usd_per_hour"]
    ):
        raise RuntimeError("RunPod allocation is outside the frozen CPU-only bounds")


def create(args: argparse.Namespace) -> None:
    plan = load_hashed(args.plan)
    if args.public_runpod_commit != plan["public_runpod_commit"]:
        raise RuntimeError("public RunPod commit mismatch")
    public_key = args.public_key.read_text(encoding="utf-8").strip()
    if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ")):
        raise RuntimeError("invalid SSH public key")
    intent = {
        "schema_version": "1.0",
        "status": "CPU-only create intent recorded before provider mutation",
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_runpod_commit,
        "compute_type": "CPU",
        "gpu_count": 0,
        "vcpu_count": plan["provider"]["vcpu_count"],
        "absolute_cap_usd": plan["budget"]["absolute_cap_usd"],
        "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
        "reserve_usd": plan["budget"]["reserve_usd"],
        "guard_seconds": plan["budget"]["guard_seconds"],
        "create_attempt_started_at": datetime.now(timezone.utc).isoformat(),
        "scientific_outputs_inspected": False,
    }
    write_hashed(args.receipt, intent)
    import requests
    from runpod_flash.core.credentials import get_api_key

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("RunPod API key is unavailable")
    provider = plan["provider"]
    body = {
        "name": provider["name"],
        "imageName": provider["image"],
        "computeType": "CPU",
        "cloudType": provider["cloud_type"],
        "cpuFlavorIds": [provider["cpu_flavor"]],
        "cpuFlavorPriority": "custom",
        "vcpuCount": provider["vcpu_count"],
        "containerDiskInGb": provider["container_disk_gb"],
        "volumeInGb": provider["persistent_volume_gb"],
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": public_key},
    }
    response = requests.post(
        "https://rest.runpod.io/v1/pods",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if response.status_code >= 400:
        intent["status"] = "provider rejected CPU-only create before billing or private transfer"
        intent["provider_http_status"] = response.status_code
        intent["provider_error_message"] = response.text[:500]
        rewrite_hashed(args.receipt, intent)
        response.raise_for_status()
    created = response.json()
    pod_id = created.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError("provider did not return a CPU Pod id")
    intent["pod_id"] = pod_id
    rewrite_hashed(args.receipt, intent)
    runpod = load_runpod()
    try:
        pod = None
        for _ in range(90):
            pod = runpod.get_pod(pod_id)
            if pod.get("desiredStatus") == "RUNNING" and float(pod.get("costPerHr") or 0.0) > 0:
                break
            time.sleep(2)
        if not pod:
            raise RuntimeError("provider did not return the created CPU Pod")
        validate_running_pod(pod, plan)
        if pod.get("desiredStatus") != "RUNNING":
            raise RuntimeError("provider did not return the CPU Pod as RUNNING")
    except Exception:
        runpod.terminate_pod(pod_id)
        intent["status"] = "created CPU Pod rejected by frozen bounds and terminated before transfer"
        intent["terminated_at"] = datetime.now(timezone.utc).isoformat()
        rewrite_hashed(args.receipt, intent)
        raise
    intent["created_at"] = datetime.now(timezone.utc).isoformat()
    intent["price_usd_per_hour"] = float(pod["costPerHr"])
    intent["observed_vcpu_count"] = int(pod["vcpuCount"])
    intent["observed_memory_gb"] = int(pod["memoryInGb"])
    intent["status"] = "RunPod CPU allocation created for sealed real scoring"
    rewrite_hashed(args.receipt, intent)
    print("RUNPOD_CPU_REAL_SCORING_CREATE_ACCEPTED")


def status(args: argparse.Namespace) -> None:
    plan = load_hashed(args.plan)
    receipt = load_hashed(args.receipt)
    if receipt["plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("receipt points to another RunPod plan")
    runpod = load_runpod()
    pod = runpod.get_pod(receipt["pod_id"])
    if pod.get("desiredStatus") == "RUNNING":
        validate_running_pod(pod, plan)
    manual = spend(receipt)
    guarded = spend(receipt, guard_seconds=receipt["guard_seconds"])
    stopped = False
    if args.enforce and guarded >= receipt["compute_cutoff_usd"] and pod.get("desiredStatus") == "RUNNING":
        runpod.stop_pod(receipt["pod_id"])
        receipt["status"] = "budget cutoff enforced; CPU allocation stopped"
        receipt["stopped_at"] = datetime.now(timezone.utc).isoformat()
        receipt["manual_spend_upper_bound_usd"] = manual
        receipt["guarded_spend_upper_bound_usd"] = guarded
        rewrite_hashed(args.receipt, receipt)
        stopped = True
    print(json.dumps({
        "provider_status": pod.get("desiredStatus"),
        "manual_spend_upper_bound_usd": manual,
        "guarded_spend_upper_bound_usd": guarded,
        "compute_cutoff_usd": receipt["compute_cutoff_usd"],
        "budget_stop_enforced": stopped,
    }, sort_keys=True))


def terminate(args: argparse.Namespace) -> None:
    receipt = load_hashed(args.receipt)
    runpod = load_runpod()
    pod = runpod.get_pod(receipt["pod_id"])
    if pod.get("desiredStatus") != "TERMINATED":
        runpod.terminate_pod(receipt["pod_id"])
    receipt["status"] = "RunPod CPU real-scoring allocation terminated after sealed copy"
    receipt["terminated_at"] = datetime.now(timezone.utc).isoformat()
    receipt["manual_spend_upper_bound_usd"] = spend(receipt)
    receipt["guarded_spend_upper_bound_usd"] = spend(
        receipt, guard_seconds=receipt["guard_seconds"]
    )
    rewrite_hashed(args.receipt, receipt)
    print("RUNPOD_CPU_REAL_SCORING_TERMINATED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "status", "terminate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-runpod-commit", default="")
    parser.add_argument("--public-key", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.command == "create" and args.public_key is None:
        parser.error("create requires --public-key")
    {"create": create, "status": status, "terminate": terminate}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
