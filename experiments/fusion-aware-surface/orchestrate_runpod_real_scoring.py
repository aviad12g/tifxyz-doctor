#!/usr/bin/env python3
"""Resume, monitor, and stop the frozen RunPod real-scoring allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def spend(receipt: dict, *, guard_seconds: int = 0) -> float:
    elapsed = max(0.0, (datetime.now(timezone.utc) - parse_time(receipt["resumed_at"])).total_seconds())
    return receipt["price_usd_per_hour"] * (elapsed + guard_seconds) / 3600.0


def load_runpod():
    import runpod

    return runpod


def validate_pod(pod: dict, plan: dict) -> None:
    expected = plan["provider"]
    observed = {
        "id": pod.get("id"),
        "machine_id": pod.get("machineId"),
        "gpu_count": pod.get("gpuCount"),
        "vcpu_count": pod.get("vcpuCount"),
        "memory_gb": pod.get("memoryInGb"),
        "price_usd_per_hour": float(pod.get("costPerHr")),
    }
    frozen = {key: expected[key] for key in observed}
    if observed != frozen:
        raise RuntimeError(f"RunPod allocation differs from frozen plan: {observed}")


def resume(args: argparse.Namespace) -> None:
    if args.receipt.exists():
        raise RuntimeError("RunPod real-scoring receipt already exists")
    plan = load_hashed(args.plan)
    if args.public_runpod_commit != plan["public_runpod_commit"]:
        raise RuntimeError("public RunPod commit mismatch")
    runpod = load_runpod()
    pod = runpod.get_pod(plan["provider"]["id"])
    validate_pod(pod, plan)
    if pod.get("desiredStatus") != "EXITED":
        raise RuntimeError("frozen RunPod allocation is not stopped")
    intent = {
        "schema_version": "1.0",
        "status": "resume intent recorded before provider mutation",
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_runpod_commit,
        "pod_id": plan["provider"]["id"],
        "price_usd_per_hour": plan["provider"]["price_usd_per_hour"],
        "absolute_cap_usd": plan["budget"]["absolute_cap_usd"],
        "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
        "reserve_usd": plan["budget"]["reserve_usd"],
        "guard_seconds": plan["budget"]["guard_seconds"],
        "resumed_at": datetime.now(timezone.utc).isoformat(),
        "scientific_outputs_inspected": False,
    }
    write_hashed(args.receipt, intent)
    runpod.resume_pod(plan["provider"]["id"], gpu_count=7)
    intent["status"] = "RunPod allocation resumed for sealed real scoring"
    write_hashed(args.receipt, intent)
    print("RUNPOD_REAL_SCORING_RESUME_ACCEPTED")


def status(args: argparse.Namespace) -> None:
    plan = load_hashed(args.plan)
    receipt = load_hashed(args.receipt)
    if receipt["plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("receipt points to another RunPod plan")
    runpod = load_runpod()
    pod = runpod.get_pod(receipt["pod_id"])
    validate_pod(pod, plan)
    manual = spend(receipt)
    guarded = spend(receipt, guard_seconds=receipt["guard_seconds"])
    stopped = False
    if args.enforce and guarded >= receipt["compute_cutoff_usd"] and pod.get("desiredStatus") == "RUNNING":
        runpod.stop_pod(receipt["pod_id"])
        receipt["status"] = "budget cutoff enforced; allocation stopped"
        receipt["stopped_at"] = datetime.now(timezone.utc).isoformat()
        receipt["manual_spend_upper_bound_usd"] = manual
        receipt["guarded_spend_upper_bound_usd"] = guarded
        write_hashed(args.receipt, receipt)
        stopped = True
    print(
        json.dumps(
            {
                "provider_status": pod.get("desiredStatus"),
                "manual_spend_upper_bound_usd": manual,
                "guarded_spend_upper_bound_usd": guarded,
                "compute_cutoff_usd": receipt["compute_cutoff_usd"],
                "budget_stop_enforced": stopped,
            },
            sort_keys=True,
        )
    )


def stop(args: argparse.Namespace) -> None:
    receipt = load_hashed(args.receipt)
    runpod = load_runpod()
    pod = runpod.get_pod(receipt["pod_id"])
    if pod.get("desiredStatus") == "RUNNING":
        runpod.stop_pod(receipt["pod_id"])
    receipt["status"] = "RunPod real-scoring allocation stopped after sealed copy"
    receipt["stopped_at"] = datetime.now(timezone.utc).isoformat()
    receipt["manual_spend_upper_bound_usd"] = spend(receipt)
    receipt["guarded_spend_upper_bound_usd"] = spend(receipt, guard_seconds=receipt["guard_seconds"])
    write_hashed(args.receipt, receipt)
    print("RUNPOD_REAL_SCORING_STOPPED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("resume", "status", "stop"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-runpod-commit", default="")
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    {"resume": resume, "status": status, "stop": stop}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
