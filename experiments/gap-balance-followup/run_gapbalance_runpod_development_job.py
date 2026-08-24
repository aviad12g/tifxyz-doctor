#!/usr/bin/env python3
"""Run one exact frozen development launcher on one isolated RTX 4090."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


PREFIX = b"# GAPBALANCE_DEVELOPMENT_JOB_CONFIG_HEX="


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


def load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    expected = body.pop("payload_sha256", None)
    if canonical(body) != expected:
        raise RuntimeError("RunPod plan payload mismatch")
    return payload


def verify_runtime() -> None:
    import torch

    expected = {
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "tifffile": "2025.2.18",
        "imagecodecs": "2024.12.30",
        "huggingface-hub": "1.11.0",
        "timm": "1.0.27",
        "einops": "0.8.1",
    }
    observed = {name: importlib.metadata.version(name).split("+", 1)[0] for name in expected}
    if observed != expected:
        raise RuntimeError(f"frozen runtime mismatch: {observed}")
    if torch.__version__.split("+", 1)[0] != "2.5.1" or str(torch.version.cuda) != "12.1":
        raise RuntimeError(f"PyTorch/CUDA mismatch: {torch.__version__}/{torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one and only one visible CUDA GPU is required")
    if "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"unexpected GPU: {torch.cuda.get_device_name(0)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    identities = {job["job_id"]: job for job in plan["jobs"]}
    raw = args.launcher.read_bytes()
    first = raw.split(b"\n", 1)[0]
    if not first.startswith(PREFIX):
        raise RuntimeError("launcher has no frozen development identity")
    embedded = json.loads(bytes.fromhex(first[len(PREFIX):].decode("ascii")))
    job_id = embedded["job_id"]
    identity = identities.get(job_id)
    if identity is None:
        raise RuntimeError(f"launcher is outside the RunPod plan: {job_id}")
    if sha256_file(args.launcher) != identity["launcher_sha256"]:
        raise RuntimeError(f"launcher identity mismatch: {job_id}")
    if embedded.get("payload_sha256") != identity["config_payload_sha256"]:
        raise RuntimeError(f"embedded config mismatch: {job_id}")
    verify_runtime()
    os.environ.update(
        KAGGLE_INPUT_PATH=str(args.input),
        KAGGLE_WORKING_PATH=str(args.working),
        KAGGLE_TEMP_PATH=str(args.temp),
    )
    spec = importlib.util.spec_from_file_location(f"runpod_{job_id.replace('-', '_')}", args.launcher)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import frozen launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_runtime = verify_runtime
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
