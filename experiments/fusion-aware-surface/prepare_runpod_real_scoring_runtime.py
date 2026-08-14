#!/usr/bin/env python3
"""Prepare the publicly frozen CPU runtime before sealed real scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_identity(path: Path, record: dict) -> None:
    if (
        path.name != record["file"]
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise RuntimeError(f"public runtime artifact identity mismatch: {path.name}")


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    subprocess.run(
        command,
        check=True,
        env=environment,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--correction-plan", type=Path, required=True)
    parser.add_argument("--uv-archive", type=Path, required=True)
    parser.add_argument("--python-archive", type=Path, required=True)
    args = parser.parse_args()

    plan = load_hashed(args.plan)
    correction = load_hashed(args.correction_plan)
    if correction["predecessor_plan_payload_sha256"] != plan["payload_sha256"]:
        raise RuntimeError("runtime correction points to another scientific plan")
    require_identity(args.uv_archive, correction["public_runtime"]["uv_archive"])
    require_identity(args.python_archive, correction["public_runtime"]["python_archive"])
    if plan["runtime"]["uv_bootstrap"] != "0.8.22":
        raise RuntimeError("unexpected frozen uv version")
    if plan["runtime"]["python"] != "3.12.13":
        raise RuntimeError("unexpected frozen CPython version")

    environment = os.environ.copy()
    environment["DEBIAN_FRONTEND"] = "noninteractive"
    run(["apt-get", "update", "-qq"], environment=environment)
    run(
        [
            "apt-get", "install", "-y", "-qq", "build-essential", "libssl-dev",
            "zlib1g-dev", "libbz2-dev", "libreadline-dev", "libsqlite3-dev",
            "libffi-dev", "liblzma-dev", "libncurses5-dev", "libncursesw5-dev",
            "libgdbm-dev", "libdb-dev", "libexpat1-dev", "tk-dev", "uuid-dev",
        ],
        environment=environment,
    )

    workspace = Path("/workspace")
    frozen_source = Path(plan["runtime"]["python_source_archive"])
    if frozen_source.exists() and sha256_file(frozen_source) != plan["runtime"]["python_source_sha256"]:
        raise RuntimeError("existing frozen CPython source archive differs")
    if not frozen_source.exists():
        shutil.copyfile(args.python_archive, frozen_source)
    if sha256_file(frozen_source) != plan["runtime"]["python_source_sha256"]:
        raise RuntimeError("frozen CPython source archive mismatch after copy")

    uv_root = workspace / "uv-0.8.22"
    uv_binary = Path("/usr/local/bin/uv")
    if not uv_binary.exists():
        if uv_root.exists():
            raise RuntimeError("unexpected pre-existing uv extraction root")
        uv_root.mkdir()
        run(["tar", "-xzf", str(args.uv_archive), "--strip-components=1", "-C", str(uv_root)])
        run(["install", "-m", "0755", str(uv_root / "uv"), str(uv_binary)])
    observed_uv = subprocess.run(
        [str(uv_binary), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if observed_uv != "uv 0.8.22":
        raise RuntimeError(f"wrong prepared uv version: {observed_uv}")

    frozen_python = Path(plan["runtime"]["python_executable"])
    if not frozen_python.exists():
        source_root = workspace / "Python-3.12.13"
        prefix = workspace / "python-3.12.13"
        if source_root.exists() or prefix.exists():
            raise RuntimeError("unexpected pre-existing CPython build or prefix")
        run(["tar", "-xzf", str(frozen_source), "-C", str(workspace)])
        run(
            [str(source_root / "configure"), f"--prefix={prefix}", "--with-ensurepip=install"],
            cwd=source_root,
        )
        run(["make", "-C", str(source_root), "-j32"])
        run(["make", "-C", str(source_root), "install"])
    observed_python = subprocess.run(
        [str(frozen_python), "-c", "import ssl,sys; print(sys.version_info[:3])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_python != "(3, 12, 13)":
        raise RuntimeError(f"wrong prepared CPython version: {observed_python}")

    marker = {
        "schema_version": "1.0",
        "state": "PUBLIC_RUNTIME_READY",
        "scientific_outputs_inspected": False,
        "plan_payload_sha256": plan["payload_sha256"],
        "correction_plan_payload_sha256": correction["payload_sha256"],
        "uv_archive_sha256": correction["public_runtime"]["uv_archive"]["sha256"],
        "python_archive_sha256": correction["public_runtime"]["python_archive"]["sha256"],
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    (workspace / "real-scoring-runtime-bootstrap.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RUNPOD_REAL_SCORING_PUBLIC_RUNTIME_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
