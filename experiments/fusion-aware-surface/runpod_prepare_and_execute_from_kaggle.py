#!/usr/bin/env python3
"""Prepare public runtime, pull sealed transport, and run the frozen real scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


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


def tree_identity(root: Path) -> dict:
    records = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"public transfer tree contains symlink: {path}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        records.append(
            f"{sha256_file(path)}  {size}  {path.relative_to(root).as_posix()}\n"
        )
        total += size
    return {
        "files": len(records),
        "bytes": total,
        "ledger_sha256": hashlib.sha256("".join(records).encode()).hexdigest(),
    }


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


def download_public(record: dict, target: Path) -> None:
    if target.exists():
        if identity(target) != {field: record[field] for field in ("file", "bytes", "sha256")}:
            raise RuntimeError(f"existing public artifact differs: {target.name}")
        return
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"stale public download exists: {partial.name}")
    request = urllib.request.Request(record["url"], headers={"User-Agent": "frozen-runpod-transport/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 << 20)
    partial.replace(target)
    if identity(target) != {field: record[field] for field in ("file", "bytes", "sha256")}:
        raise RuntimeError(f"downloaded public artifact differs: {target.name}")


def materialize_metric_source(archive: Path, target: Path) -> None:
    if target.exists():
        raise RuntimeError("public metric-source target must start absent")
    staging = target.parent / "metric-source-extract-v11"
    if staging.exists():
        raise RuntimeError("public metric-source staging must start absent")
    staging.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            pure = PurePosixPath(member.filename)
            if pure.is_absolute() or any(part in {"", ".."} for part in pure.parts):
                raise RuntimeError("public metric archive contains an unsafe path")
            mode = member.external_attr >> 16
            if mode and not (member.is_dir() or (mode & 0o170000) == 0o100000):
                raise RuntimeError("public metric archive contains a non-file member")
        bundle.extractall(staging)
    extracted = staging / "topological-metrics-kaggle"
    bundled_wheels = staging / "wheels"
    if (
        not extracted.is_dir()
        or not bundled_wheels.is_dir()
        or set(staging.iterdir()) != {extracted, bundled_wheels}
    ):
        raise RuntimeError("public metric archive layout mismatch")
    shutil.rmtree(bundled_wheels)
    extracted.rename(target)
    staging.rmdir()
    leaderboard = target / "src" / "topometrics" / "leaderboard.py"
    if sha256_file(leaderboard) != "f0db94436eea4464a30f252ebc7c35553e539da5e4832e4efaf523ed664cd811":
        raise RuntimeError("public metric leaderboard identity mismatch")


def run_logged(
    command: list[str],
    log,
    *,
    cwd: Path | None = None,
    operational_status: Path | None = None,
) -> None:
    completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        detail = ""
        if operational_status is not None and operational_status.is_file():
            child = json.loads(operational_status.read_text(encoding="utf-8"))
            if (
                child.get("state") == "ERROR"
                and child.get("scientific_outputs_inspected") is False
                and isinstance(child.get("error_type"), str)
                and isinstance(child.get("error_message"), str)
            ):
                detail = f"; {child['error_type']}: {child['error_message']}"
        raise RuntimeError(
            f"operational child failed with return code {completed.returncode}: "
            f"{Path(command[1]).name}{detail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-predecessor-plan", type=Path, required=True)
    parser.add_argument("--runtime-bootstrap-plan", type=Path, required=True)
    parser.add_argument("--runtime-preparer", type=Path, required=True)
    parser.add_argument("--puller", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--metric-verifier", type=Path, required=True)
    parser.add_argument("--metric-archive", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--pipeline-status-root", type=Path, required=True)
    parser.add_argument("--transport-status-root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--scoring-status-root", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, required=True)
    args = parser.parse_args()
    if args.pipeline_status_root.exists():
        raise RuntimeError("pipeline status root must start absent")
    args.pipeline_status_root.mkdir(parents=True)
    status_path = args.pipeline_status_root / "status.json"
    log_path = args.pipeline_status_root / "pipeline-operational.log"
    try:
        plan = load_hashed(args.plan)
        transport = plan["private_kaggle_transport"]
        if identity(Path(__file__).resolve()) != plan["private_transport_wrapper"]:
            raise RuntimeError("private transport wrapper differs from frozen plan")
        required = {
            args.runtime_preparer: plan["runtime_preparer"],
            args.runtime_bootstrap_plan: plan["runtime_bootstrap_plan"],
            args.runtime_predecessor_plan: plan["runtime_predecessor_plan"],
            args.puller: plan["private_transport_puller"],
            args.executor: plan["remote_executor"],
            args.launcher: plan["embedded_real_launcher"],
            args.metric_verifier: plan["public_metric_verifier"],
            args.metric_archive: plan["public_metric_archive"],
        }
        for path, record in required.items():
            if identity(path) != record:
                raise RuntimeError(f"public controller/artifact identity mismatch: {path.name}")
        if tree_identity(args.wheelhouse) != transport["kagglehub_wheelhouse_tree"]:
            raise RuntimeError("KaggleHub wheelhouse tree mismatch")
        if tree_identity(Path("/workspace/bundle/input/assets")) != plan["scoring_assets_tree"]:
            raise RuntimeError("public scoring-assets tree mismatch")
        if tree_identity(Path("/workspace/real-scoring-public/metric-runtime")) != plan["metric_runtime_tree"]:
            raise RuntimeError("public metric-runtime tree mismatch")
        threshold = Path("/workspace/bundle/input/threshold-freeze/frozen_thresholds.json")
        if identity(threshold) != plan["frozen_thresholds"]:
            raise RuntimeError("public frozen-threshold identity mismatch")
        write_status(status_path, "PREPARING_PUBLIC_RUNTIME", plan_payload_sha256=plan["payload_sha256"])
        bootstrap = load_hashed(args.runtime_bootstrap_plan)
        python_archive = args.plan.parent / bootstrap["public_runtime"]["python_archive"]["file"]
        uv_archive = args.plan.parent / bootstrap["public_runtime"]["uv_archive"]["file"]
        download_public(bootstrap["public_runtime"]["python_archive"], python_archive)
        download_public(bootstrap["public_runtime"]["uv_archive"], uv_archive)
        with log_path.open("ab") as log:
            run_logged(
                [
                    "python3", str(args.runtime_preparer),
                    "--plan", str(args.runtime_predecessor_plan),
                    "--correction-plan", str(args.runtime_bootstrap_plan),
                    "--uv-archive", str(uv_archive),
                    "--python-archive", str(python_archive),
                ],
                log,
            )
        materialize_metric_source(args.metric_archive, Path("/workspace/real-scoring-public/metric-source"))
        frozen_python = Path(plan["runtime"]["python_executable"])
        write_status(status_path, "PULLING_PRIVATE_TRANSPORT", plan_payload_sha256=plan["payload_sha256"])
        with log_path.open("ab") as log:
            run_logged(
                [
                    str(frozen_python), str(args.puller),
                    "--plan", str(args.plan),
                    "--python", str(frozen_python),
                    "--wheelhouse", str(args.wheelhouse),
                    "--runtime-root", "/workspace/kagglehub-runtime-v1",
                    "--credentials", str(args.credentials),
                    "--download-root", str(args.download_root),
                    "--input-root", str(args.input_root),
                    "--status-root", str(args.transport_status_root),
                ],
                log,
                operational_status=args.transport_status_root / "transport-status.json",
            )
        if not (args.transport_status_root / "PRIVATE_TRANSPORT_VERIFIED").is_file():
            raise RuntimeError("private transport verified marker is absent")
        write_status(status_path, "RUNNING_REAL_SCORER", plan_payload_sha256=plan["payload_sha256"])
        with log_path.open("ab") as log:
            run_logged(
                [
                    str(frozen_python), str(args.executor),
                    "--plan", str(args.plan),
                    "--input-manifest", str(args.input_root / "runpod_real_input_manifest.json"),
                    "--input-root", str(args.input_root),
                    "--launcher", str(args.launcher),
                    "--metric-verifier", str(args.metric_verifier),
                    "--status-root", str(args.scoring_status_root),
                    "--working-root", str(args.working_root),
                ],
                log,
                operational_status=args.scoring_status_root / "status.json",
            )
        scoring = json.loads((args.scoring_status_root / "status.json").read_text(encoding="utf-8"))
        if scoring.get("state") != "COMPLETE" or scoring.get("returncode") != 0:
            raise RuntimeError("real scorer did not leave a complete operational status")
        write_status(
            status_path,
            "COMPLETE",
            plan_payload_sha256=plan["payload_sha256"],
            returncode=0,
            transport_verified=True,
            credentials_removed=not args.credentials.exists(),
        )
        (args.pipeline_status_root / "REAL_SCORING_COMPLETE").write_text(
            plan["payload_sha256"] + "\n", encoding="utf-8"
        )
        return 0
    except Exception as error:
        if args.credentials.exists():
            args.credentials.unlink()
        write_status(
            status_path,
            "ERROR",
            error_type=type(error).__name__,
            error_message=str(error),
            operational_log=log_path.name,
            credentials_removed=not args.credentials.exists(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
