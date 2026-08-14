#!/usr/bin/env python3
"""Deploy the frozen private-Kaggle scorer to a newly created CPU Pod."""

from __future__ import annotations

import argparse
import json
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

from deploy_runpod_real_scoring_from_kaggle import (
    identity,
    load_hashed,
    run,
    tree_identity,
    validate_pod,
    write_hashed,
)


def ssh_endpoint(runpod, pod_id: str) -> tuple[str, int]:
    for _ in range(90):
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime")
        if isinstance(runtime, dict):
            for port in runtime.get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    return port["ip"], int(port["publicPort"])
        time.sleep(2)
    raise RuntimeError("new RunPod SSH endpoint did not become ready")


def validate_local_inputs(args: argparse.Namespace, plan: dict) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
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
    plan = load_hashed(args.plan)
    validate_local_inputs(args, plan)
    retry = plan.get("result_blind_verified_parallel_cpu_retry", {})
    if (
        retry.get("parallel_workers") != 32
        or retry.get("fresh_cpu_pod") is not True
        or plan["budget"].get("additional_authorized_cap_usd") != 21.0
    ):
        raise RuntimeError("fresh verified-parallel CPU retry contract mismatch")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "NEW_CPU_PRIVATE_KAGGLE_PARALLEL_PREFLIGHT_PASSED",
                    "plan_payload_sha256": plan["payload_sha256"],
                    "gpu_count": 0,
                    "parallel_workers": 32,
                    "compute_cutoff_usd": plan["budget"]["compute_cutoff_usd"],
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = load_hashed(args.receipt)
    if (
        receipt.get("plan_payload_sha256") != plan["payload_sha256"]
        or receipt.get("status") != "RunPod CPU allocation created for sealed real scoring"
        or not isinstance(receipt.get("pod_id"), str)
    ):
        raise RuntimeError("created-pod receipt differs from the frozen retry plan")
    pod_id = receipt["pod_id"]
    import runpod

    try:
        pod = runpod.get_pod(pod_id)
        validate_pod(pod)
        host, port = ssh_endpoint(runpod, pod_id)
        receipt["ssh_host"] = host
        receipt["ssh_port"] = port
        receipt["deployment_started_at"] = datetime.now(timezone.utc).isoformat()
        receipt["status"] = "fresh CPU-only private-Kaggle deployment accepted"
        write_hashed(args.receipt, receipt)
        ssh = [
            "ssh", "-p", str(port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
            f"root@{host}",
        ]
        run(
            ssh
            + [
                "command -v rsync >/dev/null || "
                "(apt-get update >/dev/null && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y rsync >/dev/null)"
            ],
            timeout=300,
        )
        roots = {
            "controller": "/workspace/real-scoring-controller-v12",
            "pipeline": "/workspace/real-scoring-pipeline-status-v12",
            "transport": "/workspace/real-transport-status-v12",
            "download": "/workspace/real-transport-download-v12",
            "input": "/workspace/real-scoring-input-kaggle-v12",
            "scoring": "/workspace/real-scoring-status-v12",
            "working": "/workspace/real-scoring-work-v12",
        }
        fixed_absent = [
            *roots.values(),
            "/workspace/kagglehub-runtime-v1",
            "/workspace/kagglehub-wheelhouse-v1",
            "/workspace/kaggle-credential-v1",
            "/workspace/real-scoring-public",
            "/workspace/bundle",
        ]
        run(ssh + [" && ".join(f"test ! -e {path}" for path in fixed_absent)], timeout=30)
        run(
            ssh + [
                f"mkdir -p {roots['controller']} /workspace/kagglehub-wheelhouse-v1 "
                "/workspace/kaggle-credential-v1 /workspace/real-scoring-public/metric-runtime "
                "/workspace/bundle/input/assets /workspace/bundle/input/threshold-freeze"
            ],
            timeout=30,
        )
        transport = (
            f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
            "-o ConnectTimeout=15"
        )
        rsync = ["rsync", "--archive", "--copy-links", "--partial", "--protect-args", "-e", transport]
        run(rsync + [str(args.wheelhouse) + "/", f"root@{host}:/workspace/kagglehub-wheelhouse-v1/"], timeout=300)
        run(rsync + [str(args.scoring_assets) + "/", f"root@{host}:/workspace/bundle/input/assets/"], timeout=300)
        run(rsync + [str(args.metric_runtime) + "/", f"root@{host}:/workspace/real-scoring-public/metric-runtime/"], timeout=900)
        controller_files = [
            args.plan, args.launcher, args.executor, args.metric_verifier, args.puller,
            args.wrapper, args.runtime_preparer, args.runtime_bootstrap_plan,
            args.runtime_predecessor_plan, args.metric_archive,
        ]
        run(rsync + [*map(str, controller_files), f"root@{host}:{roots['controller']}/"], timeout=900)
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
            f"nohup python3 {roots['controller']}/{args.wrapper.name} "
            f"--plan {roots['controller']}/{args.plan.name} "
            f"--runtime-predecessor-plan {roots['controller']}/{args.runtime_predecessor_plan.name} "
            f"--runtime-bootstrap-plan {roots['controller']}/{args.runtime_bootstrap_plan.name} "
            f"--runtime-preparer {roots['controller']}/{args.runtime_preparer.name} "
            f"--puller {roots['controller']}/{args.puller.name} "
            f"--executor {roots['controller']}/{args.executor.name} "
            f"--launcher {roots['controller']}/{args.launcher.name} "
            f"--metric-verifier {roots['controller']}/{args.metric_verifier.name} "
            f"--metric-archive {roots['controller']}/{args.metric_archive.name} "
            "--wheelhouse /workspace/kagglehub-wheelhouse-v1 "
            "--credentials /workspace/kaggle-credential-v1/kaggle.json "
            f"--pipeline-status-root {roots['pipeline']} "
            f"--transport-status-root {roots['transport']} "
            f"--download-root {roots['download']} "
            f"--input-root {roots['input']} "
            f"--scoring-status-root {roots['scoring']} "
            f"--working-root {roots['working']} "
            ">/workspace/real-scoring-bootstrap-v12.log 2>&1 </dev/null &"
        )
        run(ssh + [remote], timeout=30)
        receipt["status"] = "fresh CPU private-Kaggle verified-parallel pipeline started"
        receipt["executor_started_at"] = datetime.now(timezone.utc).isoformat()
        receipt["remote_roots"] = roots
        write_hashed(args.receipt, receipt)
        print("NEW_RUNPOD_PRIVATE_KAGGLE_PARALLEL_SCORING_ACCEPTED")
        return 0
    except Exception as error:
        try:
            runpod.stop_pod(pod_id)
        finally:
            receipt["status"] = "fresh CPU private-Kaggle deployment failed and billing was stopped"
            receipt["error_type"] = type(error).__name__
            receipt["error_message"] = str(error)[:500]
            receipt["stopped_at"] = datetime.now(timezone.utc).isoformat()
            write_hashed(args.receipt, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
