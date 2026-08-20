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


def validate_pod(
    pod: dict,
    plan: dict,
    *,
    require_exited: bool = False,
    require_exact_reused_pod: bool = True,
    expected_gpu_count: int | None = None,
) -> None:
    frozen = plan["execution"]
    if require_exact_reused_pod and (
        pod.get("id") != frozen["pod_id"] or pod.get("name") != frozen["pod_name"]
    ):
        raise RuntimeError("provider returned a different pod")
    expected_count = frozen["gpu_count"] if expected_gpu_count is None else expected_gpu_count
    if int(pod.get("gpuCount") or -1) != expected_count:
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


def terminate_many(runpod, pod_ids: list[str]) -> None:
    for pod_id in pod_ids:
        try:
            runpod.terminate_pod(pod_id)
        except Exception:
            pass


def create_equivalent_pod(runpod, *, name: str, gpu_count: int, contract: dict, plan: dict) -> dict:
    created = runpod.create_pod(
        name=name,
        image_name=contract["container_image"],
        gpu_type_id=contract["gpu_type_id"],
        cloud_type="COMMUNITY",
        support_public_ip=True,
        start_ssh=True,
        gpu_count=gpu_count,
        volume_in_gb=contract["volume_in_gb_per_pod"],
        volume_mount_path="/workspace",
        container_disk_in_gb=contract["container_disk_in_gb_per_pod"],
        min_vcpu_count=max(8, gpu_count * 4),
        min_memory_in_gb=max(32, gpu_count * 16),
        ports="22/tcp",
        env={
            "GAPBALANCE_DEVELOPMENT_PLAN_SHA256": plan["payload_sha256"],
            "GAPBALANCE_ROLE": "authoritative-development-cache-not-scored",
        },
    )
    pod_id = created["id"]
    pod = None
    for _ in range(120):
        pod = runpod.get_pod(pod_id)
        if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
            break
        time.sleep(5)
    else:
        runpod.terminate_pod(pod_id)
        raise RuntimeError(f"replacement pod did not become RUNNING: {pod_id}")
    try:
        validate_pod(
            pod,
            plan,
            require_exact_reused_pod=False,
            expected_gpu_count=gpu_count,
        )
        per_gpu = float(pod["costPerHr"]) / gpu_count
        if per_gpu > float(contract["per_gpu_hourly_ceiling_usd"]):
            raise RuntimeError(f"per-GPU provider rate exceeds public ceiling: {per_gpu}")
    except Exception:
        runpod.terminate_pod(pod_id)
        raise
    return pod


def allocate_multi(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate allocation")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    retry = load_plan(args.multi_pod_retry)
    if retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("multi-pod retry is bound to another development plan")
    if (
        retry.get("budget") != plan["budget"]
        or retry.get("authority") != plan["authority"]
        or retry.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("multi-pod retry changes a frozen budget, authority, or sealed gate")
    contract = retry["provider_contract"]
    if (
        contract["total_gpu_count"] != 7
        or contract["aggregate_hourly_ceiling_usd"] != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
        or contract["container_image"] != plan["execution"]["container_image"]
    ):
        raise RuntimeError("multi-pod provider contract differs from the primary plan")
    runpod = load_runpod()
    billing_started = now()
    errors = []
    accepted = None
    selected_layout = None
    for layout in retry["allowed_layouts_in_order"]:
        created = []
        try:
            for index, partition in enumerate(layout["partitions"]):
                pod = create_equivalent_pod(
                    runpod,
                    name=f"gapbalance-dev-{args.public_commit[:8]}-{layout['name']}-{index}",
                    gpu_count=int(partition["gpu_count"]),
                    contract=contract,
                    plan=plan,
                )
                created.append({
                    "id": pod["id"],
                    "name": pod["name"],
                    "gpu_count": int(pod["gpuCount"]),
                    "gpu_name": pod["machine"]["gpuDisplayName"],
                    "image_name": pod["imageName"],
                    "aggregate_hourly_rate_usd": float(pod["costPerHr"]),
                    "jobs": partition["jobs"],
                    "waves": partition["waves"],
                })
            rate = sum(item["aggregate_hourly_rate_usd"] for item in created)
            if rate <= 0 or rate > float(contract["aggregate_hourly_ceiling_usd"]):
                raise RuntimeError(f"aggregate provider rate exceeds public ceiling: {rate}")
            accepted = created
            selected_layout = layout["name"]
            break
        except Exception as error:
            terminate_many(runpod, [item["id"] for item in created])
            errors.append({"layout": layout["name"], "error": str(error)})
    if accepted is None:
        failure = {
            "schema_version": "1.0",
            "status": "MULTI_POD_ALLOCATION_FAILED_NO_ACTIVE_PODS",
            "plan_payload_sha256": plan["payload_sha256"],
            "multi_pod_retry_payload_sha256": retry["payload_sha256"],
            "attempt_started_at": billing_started.isoformat(),
            "layout_errors": errors,
            "scientific_endpoints_inspected": False,
            "confirmation_outputs_inspected": False,
        }
        atomic_json(args.receipt, failure)
        raise RuntimeError(f"no public multi-pod layout was available: {errors}")
    receipt = {
        "schema_version": "1.0",
        "status": "PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        "allocation_route": "equivalent_multi_pod_after_two_zero_cost_seven_gpu_failures",
        "multi_pod_retry_payload_sha256": retry["payload_sha256"],
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": billing_started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "selected_layout": selected_layout,
        "layout_attempt_errors": errors,
        "pods": accepted,
        "total_hourly_rate_usd": sum(item["aggregate_hourly_rate_usd"] for item in accepted),
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def allocate(args, plan: dict) -> None:
    if args.receipt.exists():
        raise RuntimeError("provider receipt already exists; refusing duplicate allocation")
    if len(args.public_commit) != 40 or any(c not in "0123456789abcdef" for c in args.public_commit):
        raise RuntimeError("public commit must be an exact lowercase Git SHA")
    retry = load_plan(args.allocation_retry)
    if retry.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("allocation retry is bound to another development plan")
    authorized = retry.get("authorized_retry", {})
    if (
        authorized.get("route") != "create one equivalent replacement allocation"
        or authorized.get("gpu_count") != plan["execution"]["gpu_count"]
        or authorized.get("container_image") != plan["execution"]["container_image"]
        or authorized.get("maximum_accepted_aggregate_hourly_rate_usd")
        != plan["execution"]["maximum_accepted_aggregate_hourly_rate_usd"]
        or retry.get("budget") != plan["budget"]
        or retry.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("allocation retry changes a frozen execution or budget gate")
    runpod = load_runpod()
    created = runpod.create_pod(
        name=f"gapbalance-dev-{args.public_commit[:8]}-g7",
        image_name=authorized["container_image"],
        gpu_type_id=authorized["gpu_type_id"],
        cloud_type="COMMUNITY",
        support_public_ip=True,
        start_ssh=True,
        gpu_count=authorized["gpu_count"],
        volume_in_gb=authorized["volume_in_gb"],
        volume_mount_path="/workspace",
        container_disk_in_gb=authorized["container_disk_in_gb"],
        min_vcpu_count=28,
        min_memory_in_gb=112,
        ports="22/tcp",
        env={
            "GAPBALANCE_DEVELOPMENT_PLAN_SHA256": plan["payload_sha256"],
            "GAPBALANCE_ROLE": "authoritative-development-cache-not-scored",
        },
    )
    pod_id = created["id"]
    try:
        pod = None
        for _ in range(120):
            pod = runpod.get_pod(pod_id)
            if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                break
            time.sleep(5)
        else:
            raise RuntimeError("equivalent replacement pod did not become RUNNING")
        validate_pod(pod, plan, require_exact_reused_pod=False)
    except Exception:
        runpod.terminate_pod(pod_id)
        raise
    started = now()
    receipt = {
        "schema_version": "1.0",
        "status": "POD_RESUMED_AWAITING_DEPLOYMENT",
        "allocation_route": "new_equivalent_pod_after_result_blind_resume_failure",
        "allocation_retry_payload_sha256": retry["payload_sha256"],
        "plan_payload_sha256": plan["payload_sha256"],
        "public_runpod_commit": args.public_commit,
        "billing_started_at": started.isoformat(),
        "development_billing_cutoff_usd": plan["budget"]["development_billing_cutoff_usd"],
        "absolute_campaign_cap_usd": plan["budget"]["absolute_campaign_cap_usd"],
        "confirmation_reserve_usd": plan["budget"]["confirmation_reserve_usd"],
        "pod": {
            "id": pod["id"],
            "name": pod["name"],
            "gpu_count": int(pod["gpuCount"]),
            "gpu_name": pod["machine"]["gpuDisplayName"],
            "image_name": pod["imageName"],
            "aggregate_hourly_rate_usd": float(pod["costPerHr"]),
        },
        "scientific_endpoints_inspected": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def resume_multi(args, plan: dict) -> None:
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    approval = load_plan(args.egress_resume)
    if (
        receipt.get("status") != "STOPPED"
        or receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("runpod_development_plan_payload_sha256") != plan["payload_sha256"]
        or approval.get("multi_pod_retry_payload_sha256") != receipt.get("multi_pod_retry_payload_sha256")
        or approval.get("resume", {}).get("prior_conservative_spend_usd")
        != receipt.get("conservative_development_spend_usd")
        or approval.get("budget") != plan["budget"]
        or approval.get("sealed_gates") != plan["sealed_gates"]
    ):
        raise RuntimeError("egress/resume approval is not bound to the stopped capped allocation")
    approved_pods = approval["resume"]["exact_pods_only"]
    if approved_pods != [
        {"id": pod["id"], "gpu_count": pod["gpu_count"], "jobs": pod["jobs"]}
        for pod in receipt["pods"]
    ]:
        raise RuntimeError("egress/resume approval pod set mismatch")
    runpod = load_runpod()
    resumed_ids = []
    try:
        for frozen in receipt["pods"]:
            pod = runpod.get_pod(frozen["id"])
            if pod is None or pod.get("desiredStatus") != "EXITED":
                raise RuntimeError(f"approved pod is not stopped: {frozen['id']}")
            runpod.resume_pod(frozen["id"], gpu_count=frozen["gpu_count"])
            resumed_ids.append(frozen["id"])
        for frozen in receipt["pods"]:
            for _ in range(120):
                pod = runpod.get_pod(frozen["id"])
                if pod and pod.get("desiredStatus") == "RUNNING" and pod.get("runtime"):
                    break
                time.sleep(5)
            else:
                raise RuntimeError(f"approved pod did not resume: {frozen['id']}")
            validate_pod(
                pod,
                plan,
                require_exact_reused_pod=False,
                expected_gpu_count=frozen["gpu_count"],
            )
            if float(pod["costPerHr"]) != float(frozen["aggregate_hourly_rate_usd"]):
                raise RuntimeError(f"resumed pod rate changed: {frozen['id']}")
    except Exception:
        for pod_id in resumed_ids:
            try:
                runpod.stop_pod(pod_id)
            except Exception:
                pass
        raise
    receipt.update(
        status="PODS_ALLOCATED_AWAITING_DEPLOYMENT",
        egress_resume_payload_sha256=approval["payload_sha256"],
        active_billing_started_at=now().isoformat(),
        prior_conservative_development_spend_usd=receipt["conservative_development_spend_usd"],
    )
    atomic_json(args.receipt, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "pod_ids": resumed_ids,
        "prior_conservative_development_spend_usd": receipt["prior_conservative_development_spend_usd"],
        "active_billing_started_at": receipt["active_billing_started_at"],
    }, indent=2, sort_keys=True))


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
    started = datetime.fromisoformat(receipt.get("active_billing_started_at", receipt["billing_started_at"]))
    frozen_pods = receipt.get("pods") or [receipt["pod"]]
    rate = sum(float(item["aggregate_hourly_rate_usd"]) for item in frozen_pods)
    cutoff = float(receipt["development_billing_cutoff_usd"])
    while True:
        provider = []
        any_running = False
        for frozen in frozen_pods:
            pod = runpod.get_pod(frozen["id"])
            desired = "ABSENT" if pod is None else pod.get("desiredStatus")
            any_running = any_running or desired == "RUNNING"
            provider.append({"id": frozen["id"], "desired_status": desired})
        elapsed = max(0.0, (now() - started).total_seconds())
        prior_spend = float(receipt.get("prior_conservative_development_spend_usd", 0.0))
        conservative_spend = prior_spend + rate * elapsed / 3600.0
        guarded = conservative_spend + (rate * args.guard_seconds / 3600.0 if any_running else 0.0)
        stopped = False
        if args.enforce and any_running and guarded >= cutoff:
            for frozen in frozen_pods:
                runpod.stop_pod(frozen["id"])
            stopped = True
            receipt.update(
                status="DEVELOPMENT_BUDGET_CUTOFF_STOPPED",
                budget_stop_at=now().isoformat(),
                conservative_development_spend_usd=round(conservative_spend, 6),
            )
            atomic_json(args.receipt, receipt)
        report = {
            "provider": provider,
            "elapsed_seconds": round(elapsed, 3),
            "aggregate_hourly_rate_usd": rate,
            "conservative_development_spend_usd": round(conservative_spend, 6),
            "guarded_spend_usd": round(guarded, 6),
            "development_billing_cutoff_usd": cutoff,
            "stopped_at_budget_gate": stopped,
        }
        print(json.dumps(report, sort_keys=True), flush=True)
        if stopped or not watch or not any_running:
            return
        time.sleep(args.interval)


def stop_or_terminate(args, terminate: bool) -> None:
    runpod = load_runpod()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    frozen_pods = receipt.get("pods") or [receipt["pod"]]
    for frozen in frozen_pods:
        (runpod.terminate_pod if terminate else runpod.stop_pod)(frozen["id"])
    started = datetime.fromisoformat(receipt.get("active_billing_started_at", receipt["billing_started_at"]))
    rate = sum(float(item["aggregate_hourly_rate_usd"]) for item in frozen_pods)
    spend = float(receipt.get("prior_conservative_development_spend_usd", 0.0)) + rate * max(
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
    parser.add_argument("command", choices=("resume", "resume-multi", "allocate", "allocate-multi", "status", "watch", "stop", "terminate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-commit", default="")
    parser.add_argument("--allocation-retry", type=Path)
    parser.add_argument("--multi-pod-retry", type=Path)
    parser.add_argument("--egress-resume", type=Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--guard-seconds", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.command == "resume":
        resume(args, plan)
    elif args.command == "resume-multi":
        if args.egress_resume is None:
            raise RuntimeError("resume-multi requires the public egress/resume approval")
        resume_multi(args, plan)
    elif args.command == "allocate":
        if args.allocation_retry is None:
            raise RuntimeError("allocate requires the public allocation retry")
        allocate(args, plan)
    elif args.command == "allocate-multi":
        if args.multi_pod_retry is None:
            raise RuntimeError("allocate-multi requires the public multi-pod retry")
        allocate_multi(args, plan)
    elif args.command in {"status", "watch"}:
        report_or_watch(args, plan, args.command == "watch")
    else:
        stop_or_terminate(args, args.command == "terminate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
