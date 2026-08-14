#!/usr/bin/env python3
"""Resume the sealed real-cache transfer with seven independent rsync streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_JOBS = (
    "real-test-baseline",
    "real-test-control-seed11",
    "real-test-control-seed23",
    "real-test-control-seed47",
    "real-test-gap8-seed11",
    "real-test-gap8-seed23",
    "real-test-gap8-seed47",
)


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


def run(command: list[str], *, timeout: int) -> None:
    completed = subprocess.run(command, check=False, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command[:4]}")


def validate_input_layout(root: Path) -> tuple[Path, ...]:
    manifest = root / "runpod_real_input_manifest.json"
    jobs_root = root / "jobs"
    if not manifest.is_file() or not jobs_root.is_dir():
        raise RuntimeError("sealed input root is missing its manifest or jobs directory")
    top_level = sorted(path.name for path in root.iterdir())
    if top_level != ["jobs", "runpod_real_input_manifest.json"]:
        raise RuntimeError(f"unexpected sealed input top-level entries: {top_level}")
    observed = tuple(sorted(path.name for path in jobs_root.iterdir() if path.is_dir()))
    if observed != tuple(sorted(EXPECTED_JOBS)):
        raise RuntimeError(f"unexpected sealed job directories: {observed}")
    unexpected = [path.name for path in jobs_root.iterdir() if not path.is_dir()]
    if unexpected:
        raise RuntimeError(f"unexpected non-directory job entries: {unexpected}")
    return tuple(jobs_root / name for name in EXPECTED_JOBS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--transfer-plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--public-transfer-commit", required=True)
    parser.add_argument("--sealed-input-root", type=Path, required=True)
    parser.add_argument("--scoring-assets", type=Path, required=True)
    parser.add_argument("--frozen-thresholds", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    parser.add_argument("--metric-runtime", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    args = parser.parse_args()

    plan = load_hashed(args.plan)
    transfer_plan = load_hashed(args.transfer_plan)
    receipt = load_hashed(args.receipt)
    if receipt.get("status") != "authorized sealed real-cache upload in progress":
        raise RuntimeError("receipt is not in the expected interrupted-transfer state")
    if receipt["plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("receipt points to another scientific execution plan")
    if transfer_plan["predecessor_replacement_plan_payload_sha256"] != receipt["replacement_plan"]["payload_sha256"]:
        raise RuntimeError("transfer plan points to another replacement plan")
    jobs = validate_input_layout(args.sealed_input_root)

    host = receipt["ssh_host"]
    port = int(receipt["ssh_port"])
    ssh_options = (
        f"ssh -p {port} -o BatchMode=yes -o StrictHostKeyChecking=accept-new "
        "-o ConnectTimeout=15"
    )
    ssh = [
        "ssh", "-p", str(port), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
        f"root@{host}",
    ]
    run(
        ssh + [
            "test -d /workspace/real-scoring-input && "
            "test -d /workspace/real-scoring-controller && "
            "test -d /workspace/real-scoring-public && "
            "test -d /workspace/bundle/input/assets && "
            "test -d /workspace/bundle/input/threshold-freeze && "
            "test ! -e /workspace/real-scoring-status && "
            "test ! -e /workspace/real-scoring-work && "
            "mkdir -p /workspace/real-scoring-input/jobs"
        ],
        timeout=30,
    )
    rsync_base = [
        "rsync", "--archive", "--copy-links", "--partial", "--protect-args",
        "-e", ssh_options,
    ]

    def transfer_job(job: Path) -> None:
        run(
            rsync_base + [str(job) + "/", f"root@{host}:/workspace/real-scoring-input/jobs/{job.name}/"],
            timeout=7200,
        )

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(transfer_job, job) for job in jobs]
        for future in futures:
            future.result()

    run(
        rsync_base + [
            str(args.sealed_input_root / "runpod_real_input_manifest.json"),
            f"root@{host}:/workspace/real-scoring-input/",
        ],
        timeout=300,
    )
    run(rsync_base + [str(args.scoring_assets) + "/", f"root@{host}:/workspace/bundle/input/assets/"], timeout=300)
    run(rsync_base + [str(args.frozen_thresholds), f"root@{host}:/workspace/bundle/input/threshold-freeze/"], timeout=300)
    run(rsync_base + [str(args.metric_source) + "/", f"root@{host}:/workspace/real-scoring-public/metric-source/"], timeout=900)
    run(rsync_base + [str(args.metric_runtime) + "/", f"root@{host}:/workspace/real-scoring-public/metric-runtime/"], timeout=900)
    run(
        rsync_base + [
            str(args.plan), str(args.launcher), str(args.executor),
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
    receipt.setdefault("pre_transfer_events", []).append(
        {
            "at": datetime.now(timezone.utc).isoformat(),
            "event": "seven-stream rsync completed after the single-stream path was stopped for cap protection",
            "private_payloads_opened": False,
            "scientific_outputs_inspected": False,
            "transfer_plan_payload_sha256": transfer_plan["payload_sha256"],
        }
    )
    receipt["public_transfer_commit"] = args.public_transfer_commit
    receipt["transfer_plan_payload_sha256"] = transfer_plan["payload_sha256"]
    receipt["status"] = "sealed real-scoring executor started"
    receipt["executor_started_at"] = datetime.now(timezone.utc).isoformat()
    write_hashed(args.receipt, receipt)
    print("RUNPOD_REAL_SCORING_EXECUTOR_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
