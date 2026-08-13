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


def validate_stopped_pod(pod: dict, plan: dict) -> None:
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


def validate_running_pod(pod: dict, plan: dict, gpu_count: int) -> None:
    provider = plan["provider"]
    if pod.get("id") != provider["id"] or pod.get("machineId") != provider["machine_id"]:
        raise RuntimeError("RunPod allocation moved away from the frozen pod or machine")
    if gpu_count == provider["gpu_count"]:
        minimum_vcpu = provider["vcpu_count"]
        minimum_memory = provider["memory_gb"]
        maximum_price = provider["price_usd_per_hour"]
    else:
        fallback = provider["result_blind_capacity_fallback"]
        if gpu_count != fallback["gpu_count"]:
            raise RuntimeError("requested GPU count is outside the frozen layouts")
        minimum_vcpu = fallback["minimum_vcpu_count"]
        minimum_memory = fallback["minimum_memory_gb"]
        maximum_price = fallback["maximum_price_usd_per_hour"]
    if (
        pod.get("gpuCount") != gpu_count
        or int(pod.get("vcpuCount") or 0) < minimum_vcpu
        or int(pod.get("memoryInGb") or 0) < minimum_memory
        or float(pod.get("costPerHr")) > maximum_price
    ):
        raise RuntimeError("running RunPod layout is outside the frozen capacity bounds")


def resume(args: argparse.Namespace) -> None:
    if args.receipt.exists():
        raise RuntimeError("RunPod real-scoring receipt already exists")
    plan = load_hashed(args.plan)
    if args.public_runpod_commit != plan["public_runpod_commit"]:
        raise RuntimeError("public RunPod commit mismatch")
    runpod = load_runpod()
    pod = runpod.get_pod(plan["provider"]["id"])
    validate_stopped_pod(pod, plan)
    if pod.get("desiredStatus") != "EXITED":
        raise RuntimeError("frozen RunPod allocation is not stopped")
    fallback = plan["provider"]["result_blind_capacity_fallback"]
    allowed_gpu_counts = {plan["provider"]["gpu_count"], fallback["gpu_count"]}
    if args.gpu_count not in allowed_gpu_counts:
        raise RuntimeError("requested GPU count is outside the frozen layouts")
    if args.gpu_count == fallback["gpu_count"] and not plan.get("pre_resume_capacity_event"):
        raise RuntimeError("result-blind capacity fallback lacks the frozen rejection event")
    price_upper_bound = (
        plan["provider"]["price_usd_per_hour"]
        if args.gpu_count == plan["provider"]["gpu_count"]
        else fallback["maximum_price_usd_per_hour"]
    )
    attempt_started_at = datetime.now(timezone.utc).isoformat()
    intent = {
        "schema_version": "1.0",
        "status": "resume intent recorded before provider mutation",
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_runpod_commit,
        "pod_id": plan["provider"]["id"],
        "gpu_count": args.gpu_count,
        "price_usd_per_hour": price_upper_bound,
        "absolute_cap_usd": plan["budget"]["absolute_cap_usd"],
        "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
        "reserve_usd": plan["budget"]["reserve_usd"],
        "guard_seconds": plan["budget"]["guard_seconds"],
        "resume_attempt_started_at": attempt_started_at,
        "scientific_outputs_inspected": False,
    }
    write_hashed(args.receipt, intent)
    try:
        runpod.resume_pod(plan["provider"]["id"], gpu_count=args.gpu_count)
    except Exception as error:
        intent["status"] = "provider rejected resume before billing or private transfer"
        intent["provider_error_type"] = type(error).__name__
        intent["provider_error_message"] = str(error)
        write_hashed(args.receipt, intent)
        raise
    running = runpod.get_pod(plan["provider"]["id"])
    validate_running_pod(running, plan, args.gpu_count)
    if running.get("desiredStatus") != "RUNNING":
        runpod.stop_pod(plan["provider"]["id"])
        raise RuntimeError("provider did not return the resumed allocation as RUNNING")
    intent["resumed_at"] = attempt_started_at
    intent["price_usd_per_hour"] = float(running.get("costPerHr"))
    intent["observed_vcpu_count"] = int(running.get("vcpuCount"))
    intent["observed_memory_gb"] = int(running.get("memoryInGb"))
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
    validate_running_pod(pod, plan, receipt["gpu_count"])
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
    parser.add_argument("--gpu-count", type=int, default=7)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    {"resume": resume, "status": status, "stop": stop}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
