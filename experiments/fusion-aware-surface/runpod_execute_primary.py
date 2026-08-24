#!/usr/bin/env python3
"""Remote executor for seven sealed, frozen RunPod synthetic jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("payload_sha256")
    body = dict(payload)
    body.pop("payload_sha256", None)
    observed = sha256_bytes(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed != expected:
        raise RuntimeError(f"replacement plan payload mismatch: {observed}")
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_bundle(root: Path, manifest_path: Path, plan: dict) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("payload_sha256")
    body = dict(manifest)
    body.pop("payload_sha256", None)
    if sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()) != expected:
        raise RuntimeError("bundle manifest payload mismatch")
    if manifest.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("bundle is bound to another replacement plan")
    expected_paths = set()
    for record in manifest.get("files", []):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe bundle path: {relative}")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"bundle file identity mismatch: {relative}")
        expected_paths.add(relative.as_posix())
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("bundle file-set mismatch")
    return manifest


def install_runtime() -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "numpy==1.26.4",
            "scipy==1.16.3",
            "tifffile==2025.2.18",
            "imagecodecs==2024.12.30",
            "huggingface-hub==1.11.0",
            "timm==1.0.27",
            "einops==0.8.1",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def smoke_gpus(worker_count: int) -> dict:
    import torch

    if torch.__version__.split("+", 1)[0] != "2.5.1" or str(torch.version.cuda) != "12.1":
        raise RuntimeError(f"frozen torch runtime mismatch: {torch.__version__}/{torch.version.cuda}")
    visible = torch.cuda.device_count()
    allowed = {1} if worker_count == 1 else {7, 8}
    if visible not in allowed or visible < worker_count:
        raise RuntimeError(f"unexpected visible GPU count for {worker_count} workers: {visible}")
    names = []
    for index in range(worker_count):
        name = torch.cuda.get_device_name(index)
        if "RTX 4090" not in name:
            raise RuntimeError(f"GPU {index} is not an RTX 4090: {name}")
        with torch.cuda.device(index):
            value = torch.ones((16, 16), device=f"cuda:{index}")
            result = value @ value
            torch.cuda.synchronize(index)
            if result.shape != (16, 16) or not bool(torch.isfinite(result).all()):
                raise RuntimeError(f"GPU {index} operational smoke failed")
        names.append(name)
    return {"gpu_names": names, "visible_gpu_count": visible}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--launchers", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()
    plan = load_plan(args.plan)
    bundle = verify_bundle(args.bundle_root, args.bundle_manifest, plan)
    all_jobs = [item["job_id"] for item in plan["jobs"]]
    jobs = args.job_id or all_jobs
    if any(job not in all_jobs for job in jobs) or len(set(jobs)) != len(jobs):
        raise RuntimeError("invalid or duplicate primary job selection")
    if jobs != bundle.get("job_ids"):
        raise RuntimeError("executor job selection does not match staged bundle")
    if len(jobs) not in {1, 7}:
        raise RuntimeError("executor permits one isolated job or the exact seven-job primary")
    args.working.mkdir(parents=True, exist_ok=True)
    args.temp.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    if args.status.exists():
        raise RuntimeError("operational status must start absent")
    launchers = {
        job: args.launchers / job / "heldout_cache_launcher.py" for job in jobs
    }
    missing = [str(path) for path in launchers.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"frozen launchers absent: {missing}")

    state = {
        "jobs": {job: {"state": "WAITING"} for job in jobs},
        "plan_payload_sha256": plan["payload_sha256"],
        "scientific_outputs_inspected": False,
        "state": "INSTALLING_RUNTIME",
    }
    atomic_json(args.status, state)
    install_runtime()
    smoke = smoke_gpus(len(jobs))
    state.update(state="RUNNING", runtime_smoke=smoke)
    processes = {}
    logs = {}
    for gpu_index, job in enumerate(jobs):
        job_working = args.working / job
        job_temp = args.temp / job
        job_working.mkdir()
        job_temp.mkdir()
        log_path = args.status.parent / f"{job}.operational.log"
        log = log_path.open("wb")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        command = [
            sys.executable,
            str(args.wrapper),
            "--launcher",
            str(launchers[job]),
            "--plan",
            str(args.plan),
            "--input",
            str(args.input),
            "--working",
            str(job_working),
            "--temp",
            str(job_temp),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes[job] = process
        logs[job] = log
        state["jobs"][job] = {
            "gpu_index": gpu_index,
            "operational_log": log_path.name,
            "pid": process.pid,
            "state": "RUNNING",
        }
    atomic_json(args.status, state)

    failed = None
    while processes:
        for job, process in list(processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            logs[job].close()
            state["jobs"][job]["returncode"] = returncode
            state["jobs"][job]["state"] = "COMPLETE" if returncode == 0 else "ERROR"
            processes.pop(job)
            if returncode != 0 and failed is None:
                failed = job
        if failed is not None:
            for job, process in processes.items():
                process.terminate()
                state["jobs"][job]["state"] = "TERMINATING_AFTER_PEER_ERROR"
            atomic_json(args.status, state)
            for job, process in processes.items():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                logs[job].close()
                state["jobs"][job]["returncode"] = process.returncode
                state["jobs"][job]["state"] = "ABORTED_AFTER_PEER_ERROR"
            state["state"] = "ERROR"
            state["failed_job"] = failed
            atomic_json(args.status, state)
            return 1
        atomic_json(args.status, state)
        if processes:
            time.sleep(5)

    state["state"] = "COMPLETE"
    state["scientific_outputs_inspected"] = False
    atomic_json(args.status, state)
    (args.status.parent / "PRIMARY_COMPLETE").write_text(
        plan["payload_sha256"] + "\n", encoding="utf-8"
    )
    print("RUNPOD_PRIMARY_SEVEN_JOBS_COMPLETE_SEALED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
