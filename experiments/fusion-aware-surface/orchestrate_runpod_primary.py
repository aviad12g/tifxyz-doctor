#!/usr/bin/env python3
"""Provider controller and hard budget gate for the RunPod primary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    observed = sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    if observed != expected:
        raise RuntimeError(f"replacement plan payload mismatch: {observed}")
    return payload


def load_runpod():
    try:
        import runpod
    except ImportError as error:
        raise RuntimeError("install the frozen controller dependency runpod==1.9.0") from error
    return runpod


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def clean_pod(pod: dict, jobs: list[str]) -> dict:
    return {
        "accepted_at": now(),
        "cost_per_hour_usd": float(pod["costPerHr"]),
        "gpu_count": int(pod["gpuCount"]),
        "gpu_name": pod["machine"]["gpuDisplayName"],
        "id": pod["id"],
        "image_name": pod["imageName"],
        "jobs": jobs,
        "name": pod["name"],
    }


def validate_pod(pod: dict, plan: dict, expected_gpu_count: int) -> None:
    execution = plan["execution"]
    if int(pod["gpuCount"]) != expected_gpu_count:
        raise RuntimeError(f"GPU count mismatch for {pod['id']}")
    if "RTX 4090" not in pod["machine"]["gpuDisplayName"]:
        raise RuntimeError(f"wrong GPU for {pod['id']}: {pod['machine']['gpuDisplayName']}")
    if pod["imageName"] != plan["container"]["image"]:
        raise RuntimeError(f"container tag mismatch for {pod['id']}")
    per_gpu = float(pod["costPerHr"]) / expected_gpu_count
    if per_gpu > plan["budget"]["maximum_accepted_gpu_hourly_price_usd"]:
        raise RuntimeError(f"GPU price ${per_gpu:.4f}/h exceeds frozen ceiling")
    if execution["cloud_type"] != "COMMUNITY":
        raise RuntimeError("unexpected frozen cloud type")


def create_one(runpod, plan: dict, gpu_count: int, name: str) -> dict:
    created = runpod.create_pod(
        name=name,
        image_name=plan["container"]["image"],
        gpu_type_id=plan["execution"]["gpu_type_id"],
        cloud_type=plan["execution"]["cloud_type"],
        support_public_ip=True,
        start_ssh=True,
        gpu_count=gpu_count,
        volume_in_gb=25,
        volume_mount_path="/workspace",
        container_disk_in_gb=30,
        min_vcpu_count=max(8, gpu_count * 4),
        min_memory_in_gb=max(32, gpu_count * 16),
        ports="22/tcp",
        env={
            "RUNPOD_PRIMARY_PLAN_PAYLOAD_SHA256": plan["payload_sha256"],
            "RUNPOD_PRIMARY_ROLE": "authoritative-primary-sealed",
        },
    )
    pod_id = created["id"]
    try:
        for _ in range(30):
            pod = runpod.get_pod(pod_id)
            if pod and pod.get("machine") and pod.get("costPerHr") is not None:
                return pod
            time.sleep(2)
    except Exception:
        runpod.terminate_pod(pod_id)
        raise
    runpod.terminate_pod(pod_id)
    raise RuntimeError(f"created pod never became queryable: {pod_id}")


def terminate_many(runpod, pod_ids: list[str]) -> None:
    for pod_id in pod_ids:
        try:
            runpod.terminate_pod(pod_id)
        except Exception:
            pass


def launch(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("RunPod receipt already exists; refusing duplicate launch")
    if len(args.replacement_commit) != 40:
        raise RuntimeError("public replacement commit must be a full 40-character commit")
    runpod = load_runpod()
    jobs = [item["job_id"] for item in plan["jobs"]]
    accepted = []
    errors = []
    layouts = plan["execution"]["allowed_layouts_in_order"]
    for layout in layouts:
        pod_count = layout["pod_count"]
        gpu_count = layout["gpu_count_per_pod"]
        candidate = []
        created_ids = []
        try:
            if pod_count == 1:
                pod = create_one(runpod, plan, gpu_count, f"vesuvius-synth-primary-{args.replacement_commit[:8]}-g{gpu_count}")
                created_ids.append(pod["id"])
                validate_pod(pod, plan, gpu_count)
                candidate.append(clean_pod(pod, jobs))
            elif pod_count == 7 and gpu_count == 1:
                for index, job in enumerate(jobs):
                    pod = create_one(runpod, plan, 1, f"vesuvius-synth-{index}-{args.replacement_commit[:8]}")
                    created_ids.append(pod["id"])
                    validate_pod(pod, plan, 1)
                    candidate.append(clean_pod(pod, [job]))
            else:
                raise RuntimeError(f"unsupported frozen layout: {layout}")
            accepted = candidate
            selected = layout
            break
        except Exception as error:
            terminate_many(runpod, created_ids)
            errors.append({"layout": layout, "error": str(error)})
    if not accepted:
        raise RuntimeError(f"no frozen RTX 4090 layout was available: {errors}")
    total_rate = sum(item["cost_per_hour_usd"] for item in accepted)
    if total_rate <= 0 or total_rate > 8 * plan["budget"]["maximum_accepted_gpu_hourly_price_usd"]:
        terminate_many(runpod, [item["id"] for item in accepted])
        raise RuntimeError(f"invalid aggregate hourly rate: {total_rate}")
    receipt = {
        "billing_cutoff_usd": plan["budget"]["billing_cutoff_usd"],
        "created_at": now(),
        "hard_total_cap_usd": plan["budget"]["hard_total_cap_usd"],
        "layout": selected,
        "layout_attempt_errors": errors,
        "plan_payload_sha256": plan["payload_sha256"],
        "pods": accepted,
        "public_replacement_commit": args.replacement_commit,
        "role": "authoritative primary; Kaggle is sealed secondary replication",
        "scientific_outputs_inspected": False,
        "status": "PODS_ACCEPTED_AWAITING_STAGING",
        "total_hourly_rate_usd": total_rate,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def status(args, plan: dict, watch: bool = False) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt["plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("provider receipt is bound to another plan")
    while True:
        provider = []
        spend = 0.0
        active_rate = 0.0
        for frozen in receipt["pods"]:
            pod = runpod.get_pod(frozen["id"])
            if pod is None:
                provider.append({"id": frozen["id"], "state": "ABSENT"})
                continue
            uptime = float(pod.get("uptimeSeconds") or 0)
            rate = float(pod.get("costPerHr") or frozen["cost_per_hour_usd"])
            spend += rate * uptime / 3600.0
            if pod.get("desiredStatus") == "RUNNING":
                active_rate += rate
            provider.append({
                "cost_per_hour_usd": rate,
                "desired_status": pod.get("desiredStatus"),
                "id": pod["id"],
                "uptime_seconds": uptime,
            })
        projected = spend + active_rate * args.guard_seconds / 3600.0
        stopped = []
        if args.enforce and projected >= receipt["billing_cutoff_usd"]:
            for item in provider:
                if item.get("desired_status") == "RUNNING":
                    runpod.stop_pod(item["id"])
                    stopped.append(item["id"])
            receipt["status"] = "BUDGET_CUTOFF_STOPPED"
            receipt["budget_stop_at"] = now()
            receipt["budget_stop_projected_compute_usd"] = projected
            atomic_json(args.receipt, receipt)
        report = {
            "active_hourly_rate_usd": active_rate,
            "billing_cutoff_usd": receipt["billing_cutoff_usd"],
            "compute_spend_usd": round(spend, 6),
            "guard_seconds": args.guard_seconds,
            "projected_guarded_spend_usd": round(projected, 6),
            "provider": provider,
            "stopped_at_budget_gate": stopped,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if stopped or not watch or active_rate == 0:
            return
        time.sleep(args.interval)


def stop_or_terminate(args, terminate: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    for pod in receipt["pods"]:
        (runpod.terminate_pod if terminate else runpod.stop_pod)(pod["id"])
    receipt["status"] = "TERMINATED" if terminate else "STOPPED"
    receipt["provider_action_at"] = now()
    atomic_json(args.receipt, receipt)
    print(receipt["status"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "status", "watch", "stop", "terminate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--replacement-commit", default="")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--guard-seconds", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.command == "launch":
        launch(args, plan)
    elif args.command in {"status", "watch"}:
        status(args, plan, watch=args.command == "watch")
    elif args.command == "stop":
        stop_or_terminate(args, terminate=False)
    else:
        stop_or_terminate(args, terminate=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
