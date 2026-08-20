#!/usr/bin/env python3
"""Resume, monitor, and stop the exact $25-capped RunPod campaign pod."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def now() -> datetime:
    return datetime.now(timezone.utc)


def canonical(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError("RunPod plan payload mismatch")
    return payload


def load_runpod():
    try:
        import runpod
    except ImportError as error:
        raise RuntimeError("controller requires runpod==1.9.0") from error
    return runpod


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def validate_pod(pod: dict, plan: dict, *, require_exited: bool = False) -> None:
    frozen = plan["execution"]
    if pod.get("id") != frozen["pod_id"] or pod.get("name") != frozen["pod_name"]:
        raise RuntimeError("provider returned a different pod")
    if int(pod.get("gpuCount") or -1) != frozen["gpu_count"]:
        raise RuntimeError("provider GPU count mismatch")
    if "RTX 4090" not in (pod.get("machine") or {}).get("gpuDisplayName", ""):
        raise RuntimeError("provider GPU type mismatch")
    if pod.get("imageName") != frozen["container_image"]:
        raise RuntimeError("provider image mismatch")
    rate = float(pod.get("costPerHr") or 0)
    if rate <= 0 or rate > frozen["maximum_accepted_aggregate_hourly_rate_usd"]:
        raise RuntimeError(f"provider rate outside frozen ceiling: {rate}")
    if require_exited and pod.get("desiredStatus") != "EXITED":
        raise RuntimeError("exact reusable pod is not stopped")


def resume(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate resume")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    runpod = load_runpod()
    pod_id = plan["execution"]["pod_id"]
    pod = runpod.get_pod(pod_id)
    validate_pod(pod, plan, require_exited=True)
    runpod.resume_pod(pod_id, gpu_count=plan["execution"]["gpu_count"])
    resumed = None
    for _ in range(120):
        resumed = runpod.get_pod(pod_id)
        if resumed and resumed.get("desiredStatus") == "RUNNING" and resumed.get("runtime"):
            break
        time.sleep(5)
    else:
        runpod.stop_pod(pod_id)
        raise RuntimeError("exact RunPod pod did not become RUNNING")
    validate_pod(resumed, plan)
    started = now()
    receipt = {
        "schema_version": "1.0",
        "status": "POD_RESUMED_AWAITING_DEPLOYMENT",
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "pod": {
            "id": resumed["id"],
            "name": resumed["name"],
            "gpu_count": int(resumed["gpuCount"]),
            "gpu_name": resumed["machine"]["gpuDisplayName"],
            "image_name": resumed["imageName"],
            "aggregate_hourly_rate_usd": float(resumed["costPerHr"]),
        },
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def report_or_watch(args, plan: dict, watch: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("receipt is bound to another RunPod plan")
    started = datetime.fromisoformat(receipt["billing_started_at"])
    rate = float(receipt["pod"]["aggregate_hourly_rate_usd"])
    cutoff = float(receipt["development_billing_cutoff_usd"])
    while True:
        pod = runpod.get_pod(receipt["pod"]["id"])
        desired = "ABSENT" if pod is None else pod.get("desiredStatus")
        elapsed = max(0.0, (now() - started).total_seconds())
        conservative_spend = rate * elapsed / 3600.0
        guarded = conservative_spend + (rate * args.guard_seconds / 3600.0 if desired == "RUNNING" else 0.0)
        stopped = False
        if args.enforce and desired == "RUNNING" and guarded >= cutoff:
            runpod.stop_pod(receipt["pod"]["id"])
            stopped = True
            receipt.update(
                status="DEVELOPMENT_BUDGET_CUTOFF_STOPPED",
                budget_stop_at=now().isoformat(),
                conservative_development_spend_usd=round(conservative_spend, 6),
            )
            atomic_json(args.receipt, receipt)
        report = {
            "pod_id": receipt["pod"]["id"],
            "desired_status": desired,
            "elapsed_seconds": round(elapsed, 3),
            "aggregate_hourly_rate_usd": rate,
            "conservative_development_spend_usd": round(conservative_spend, 6),
            "guarded_spend_usd": round(guarded, 6),
            "development_billing_cutoff_usd": cutoff,
            "stopped_at_budget_gate": stopped,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if stopped or not watch or desired != "RUNNING":
            return
        time.sleep(args.interval)


def stop_or_terminate(args, terminate: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    pod_id = receipt["pod"]["id"]
    (runpod.terminate_pod if terminate else runpod.stop_pod)(pod_id)
    started = datetime.fromisoformat(receipt["billing_started_at"])
    spend = float(receipt["pod"]["aggregate_hourly_rate_usd"]) * max(
        0.0, (now() - started).total_seconds()
    ) / 3600.0
    receipt.update(
        status="TERMINATED" if terminate else "STOPPED",
        provider_action_at=now().isoformat(),
        conservative_development_spend_usd=round(spend, 6),
    )
    atomic_json(args.receipt, receipt)
    print(receipt["status"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("resume", "status", "watch", "stop", "terminate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-commit", default="")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--guard-seconds", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.command == "resume":
        resume(args, plan)
    elif args.command in {"status", "watch"}:
        report_or_watch(args, plan, args.command == "watch")
    else:
        stop_or_terminate(args, args.command == "terminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
