#!/usr/bin/env python3
"""Execute the frozen real one-shot scorer on the CPU-only RunPod allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def identity(path: Path) -> dict:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_status(path: Path, state: str, **extra: object) -> None:
    payload = {
        "schema_version": "1.0",
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_outputs_inspected": False,
        **extra,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_sealed_inputs(root: Path, manifest: dict) -> None:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 7:
        raise RuntimeError("sealed real-input job count mismatch")
    files = 0
    total = 0
    for job in jobs:
        job_root = root / "jobs" / job["job_id"]
        for key in ("job_index", "cache_manifest"):
            expected = job[key]
            path = job_root / expected["file"]
            if identity(path) != {field: expected[field] for field in ("file", "bytes", "sha256")}:
                raise RuntimeError(f"{job['job_id']}: {key} identity mismatch")
        expected_names = set()
        for expected in job["sealed_cache_files"]:
            path = job_root / "sealed-caches" / expected["file"]
            if identity(path) != expected:
                raise RuntimeError(f"{job['job_id']}: sealed cache identity mismatch")
            expected_names.add(expected["file"])
            files += 1
            total += expected["bytes"]
        observed_names = {path.name for path in (job_root / "sealed-caches").iterdir() if path.is_file()}
        if observed_names != expected_names:
            raise RuntimeError(f"{job['job_id']}: sealed cache file set mismatch")
    if files != 266 or total != 5_791_045_122:
        raise RuntimeError("sealed real-input aggregate mismatch")


def materialize_scoring_layout(root: Path, manifest: dict) -> None:
    """Expose each sealed cache directory under its frozen run name.

    The transport bundle deliberately stores every NPZ under a uniform
    ``sealed-caches`` directory.  The public scorer's result-blind stager
    expects the original run directory recorded in each held-out job index.
    A relative directory symlink provides that exact view without copying or
    opening any NPZ payload and without creating duplicate job indexes.
    """
    for job in manifest["jobs"]:
        job_root = root / "jobs" / job["job_id"]
        index = load_hashed(job_root / job["job_index"]["file"])
        run = index.get("job", {}).get("run")
        if not isinstance(run, str) or not run or Path(run).name != run:
            raise RuntimeError(f"{job['job_id']}: invalid frozen run directory")
        sealed = job_root / "sealed-caches"
        view = job_root / run
        if view.exists() or view.is_symlink():
            if not view.is_symlink() or view.resolve() != sealed.resolve():
                raise RuntimeError(f"{job['job_id']}: existing scoring run view identity mismatch")
        else:
            view.symlink_to(sealed.name, target_is_directory=True)
        if not view.is_dir() or view.resolve() != sealed.resolve():
            raise RuntimeError(f"{job['job_id']}: scoring run view identity mismatch")


def materialize_public_metric_layout() -> None:
    source = Path("/workspace/real-scoring-public/metric-source")
    view = Path("/workspace/topological-metrics-kaggle")
    if not source.is_dir():
        raise RuntimeError("pinned public metric source directory is absent")
    if view.exists() or view.is_symlink():
        if not view.is_symlink() or view.resolve() != source.resolve():
            raise RuntimeError("existing public metric source view differs")
    else:
        view.symlink_to(source, target_is_directory=True)
    if not view.is_dir() or view.resolve() != source.resolve():
        raise RuntimeError("public metric source view mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--status-root", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    args = parser.parse_args()
    args.status_root.mkdir(parents=True, exist_ok=True)
    status_path = args.status_root / "status.json"
    log_path = args.status_root / "real-scoring-operational.log"
    try:
        plan = load_hashed(args.plan)
        manifest = load_hashed(args.input_manifest)
        if plan.get("status") != "result-blind CPU-only RunPod real one-shot scoring frozen before private cache egress":
            raise RuntimeError("wrong CPU-only RunPod real-scoring plan status")
        if identity(Path(__file__).resolve()) != plan["remote_executor"]:
            raise RuntimeError("remote executor differs from frozen plan")
        if identity(args.launcher) != plan["embedded_real_launcher"]:
            raise RuntimeError("embedded real launcher differs from frozen plan")
        observed_manifest_identity = identity(args.input_manifest)
        observed_manifest_identity["payload_sha256"] = manifest["payload_sha256"]
        if observed_manifest_identity != plan["sealed_real_inputs"]:
            raise RuntimeError("sealed real-input manifest differs from frozen plan")
        write_status(status_path, "VERIFYING_SEALED_INPUTS", plan_payload_sha256=plan["payload_sha256"])
        verify_sealed_inputs(args.input_root, manifest)
        materialize_scoring_layout(args.input_root, manifest)
        materialize_public_metric_layout()
        assets = Path("/workspace/bundle/input/assets/SOURCE_SHA256SUMS")
        threshold = Path("/workspace/bundle/input/threshold-freeze/frozen_thresholds.json")
        if sha256_file(assets) != plan["scoring_assets"]["ledger_sha256"]:
            raise RuntimeError("minimal scoring asset ledger identity mismatch")
        if sha256_file(threshold) != plan["frozen_thresholds"]["sha256"]:
            raise RuntimeError("frozen threshold identity mismatch")
        write_status(status_path, "INSTALLING_RUNTIME", plan_payload_sha256=plan["payload_sha256"])
        args.working_root.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("frozen uv bootstrap executable is absent")
        observed_uv = subprocess.run(
            [uv, "--version"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if observed_uv != "uv 0.8.22":
            raise RuntimeError(f"wrong uv bootstrap runtime: {observed_uv}")
        frozen_python = Path(plan["runtime"]["python_executable"])
        if not frozen_python.is_file():
            raise RuntimeError("frozen CPython executable is absent")
        observed_source = sha256_file(Path(plan["runtime"]["python_source_archive"]))
        if observed_source != plan["runtime"]["python_source_sha256"]:
            raise RuntimeError("frozen CPython source archive identity mismatch")
        observed_base = subprocess.run(
            [str(frozen_python), "-c", "import sys; print(sys.version_info[:3])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_base != "(3, 12, 13)":
            raise RuntimeError(f"wrong frozen CPython runtime: {observed_base}")
        environment_root = args.working_root / "venv"
        subprocess.run(
            [
                uv,
                "venv",
                "--seed",
                "--python",
                str(frozen_python),
                str(environment_root),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        python = environment_root / "bin" / "python"
        observed = subprocess.run(
            [str(python), "-c", "import sys; print(sys.version_info[:3])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed != "(3, 12, 13)":
            raise RuntimeError(f"wrong scoring Python runtime: {observed}")
        kaggle_working = args.working_root / "kaggle-working"
        kaggle_temp = args.working_root / "kaggle-temp"
        kaggle_working.mkdir()
        kaggle_temp.mkdir()
        environment = os.environ.copy()
        environment.update(
            {
                "KAGGLE_INPUT_PATH": "/workspace",
                "KAGGLE_WORKING_PATH": str(kaggle_working),
                "KAGGLE_TEMP_PATH": str(kaggle_temp),
                "PYTHONUNBUFFERED": "1",
            }
        )
        write_status(status_path, "RUNNING", plan_payload_sha256=plan["payload_sha256"])
        with log_path.open("wb") as log:
            completed = subprocess.run(
                [str(python), str(args.launcher)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"real one-shot scorer failed with return code {completed.returncode}")
        output = kaggle_working / "fusion-one-shot-real"
        expected = {
            "sealed_real_test_results.json",
            "scoring_run_manifest.json",
            "real-panels",
        }
        if {path.name for path in output.iterdir()} != expected:
            raise RuntimeError("real one-shot output file set mismatch")
        write_status(
            status_path,
            "COMPLETE",
            plan_payload_sha256=plan["payload_sha256"],
            output_root=str(output),
            returncode=0,
        )
        (args.status_root / "REAL_SCORING_COMPLETE").write_text(
            plan["payload_sha256"] + "\n", encoding="utf-8"
        )
        return 0
    except Exception as error:
        write_status(
            status_path,
            "ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
            operational_log=log_path.name,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
