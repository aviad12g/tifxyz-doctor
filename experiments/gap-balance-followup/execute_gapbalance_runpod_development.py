#!/usr/bin/env python3
"""Remote two-wave executor for the frozen 12-job development replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_bundle(root: Path, manifest_path: Path, plan: dict) -> dict:
    manifest = load_hashed(manifest_path)
    if manifest.get("plan_payload_sha256") != plan["payload_sha256"]:
        raise RuntimeError("bundle is bound to another RunPod plan")
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
            raise RuntimeError(f"bundle identity mismatch: {relative}")
        expected_paths.add(relative.as_posix())
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if observed_paths != expected_paths:
        raise RuntimeError("bundle file-set mismatch")
    if manifest.get("scientific_outputs_present") is not False:
        raise RuntimeError("input bundle violates the blind gate")
    return manifest


def install_runtime() -> None:
    commands = [
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
            "torch==2.5.1", "torchvision==0.20.1", "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
            "numpy==1.26.4", "scipy==1.16.3", "tifffile==2025.2.18",
            "imagecodecs==2024.12.30", "huggingface-hub==1.11.0",
            "timm==1.0.27", "einops==0.8.1",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def smoke_gpus(plan: dict) -> dict:
    import torch

    expected = int(plan["execution"]["gpu_count"])
    if torch.__version__.split("+", 1)[0] != "2.5.1" or str(torch.version.cuda) != "12.1":
        raise RuntimeError(f"frozen torch mismatch: {torch.__version__}/{torch.version.cuda}")
    if torch.cuda.device_count() != expected:
        raise RuntimeError(f"expected exactly {expected} visible GPUs")
    names = []
    for index in range(expected):
        name = torch.cuda.get_device_name(index)
        if "RTX 4090" not in name:
            raise RuntimeError(f"GPU {index} is not RTX 4090: {name}")
        with torch.cuda.device(index):
            value = torch.ones((16, 16), device=f"cuda:{index}")
            result = value @ value
            torch.cuda.synchronize(index)
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError(f"GPU {index} smoke failed")
        names.append(name)
    return {"visible_gpu_count": expected, "gpu_names": names}


def run_wave(
    *,
    wave_index: int,
    jobs: list[str],
    launchers: Path,
    wrapper: Path,
    plan_path: Path,
    input_root: Path,
    working: Path,
    temp: Path,
    status_path: Path,
    state: dict,
) -> bool:
    processes: dict[str, subprocess.Popen] = {}
    logs = {}
    for gpu_index, job in enumerate(jobs):
        job_working = working / job
        job_temp = temp / job
        if job_working.exists() or job_temp.exists():
            raise RuntimeError(f"job paths must start absent: {job}")
        job_working.mkdir(parents=True)
        job_temp.mkdir(parents=True)
        log_path = status_path.parent / f"{job}.operational.log"
        log = log_path.open("wb")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        command = [
            sys.executable,
            str(wrapper),
            "--launcher", str(launchers / job / "gapbalance_development_kaggle_launcher.py"),
            "--plan", str(plan_path),
            "--input", str(input_root),
            "--working", str(job_working),
            "--temp", str(job_temp),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        processes[job] = process
        logs[job] = log
        state["jobs"][job] = {
            "gpu_index": gpu_index,
            "operational_log": log_path.name,
            "pid": process.pid,
            "state": "RUNNING",
            "wave": wave_index,
        }
    state["active_wave"] = wave_index
    atomic_json(status_path, state)
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
            atomic_json(status_path, state)
            for job, process in processes.items():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logs[job].close()
                state["jobs"][job]["returncode"] = process.returncode
                state["jobs"][job]["state"] = "ABORTED_AFTER_PEER_ERROR"
            state.update(state="ERROR", failed_job=failed)
            atomic_json(status_path, state)
            return False
        atomic_json(status_path, state)
        if processes:
            time.sleep(5)
    return True


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
    args = parser.parse_args()
    plan = load_hashed(args.plan)
    bundle = verify_bundle(args.bundle_root, args.bundle_manifest, plan)
    job_ids = [job["job_id"] for job in plan["jobs"]]
    waves = plan["execution"]["waves"]
    if [job for wave in waves for job in wave] != job_ids or [len(wave) for wave in waves] != [7, 5]:
        raise RuntimeError("RunPod wave layout mismatch")
    if bundle.get("job_ids") != job_ids:
        raise RuntimeError("bundle job list mismatch")
    if args.status.exists() or args.working.exists() or args.temp.exists():
        raise RuntimeError("remote status/output/temp roots must start absent")
    args.working.mkdir(parents=True)
    args.temp.mkdir(parents=True)
    state = {
        "state": "INSTALLING_RUNTIME",
        "plan_payload_sha256": plan["payload_sha256"],
        "jobs": {job: {"state": "WAITING"} for job in job_ids},
        "scientific_endpoints_scored": False,
        "confirmation_outputs_inspected": False,
    }
    atomic_json(args.status, state)
    install_runtime()
    state.update(state="RUNNING", runtime_smoke=smoke_gpus(plan))
    atomic_json(args.status, state)
    for wave_index, jobs in enumerate(waves):
        if not run_wave(
            wave_index=wave_index,
            jobs=jobs,
            launchers=args.launchers,
            wrapper=args.wrapper,
            plan_path=args.plan,
            input_root=args.input,
            working=args.working,
            temp=args.temp,
            status_path=args.status,
            state=state,
        ):
            return 1
    state.update(state="COMPLETE", active_wave=None)
    atomic_json(args.status, state)
    (args.status.parent / "PRIMARY_COMPLETE").write_text(
        plan["payload_sha256"] + "\n", encoding="utf-8"
    )
    print("GAPBALANCE_RUNPOD_12_DEVELOPMENT_JOBS_COMPLETE_NOT_SCORED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
