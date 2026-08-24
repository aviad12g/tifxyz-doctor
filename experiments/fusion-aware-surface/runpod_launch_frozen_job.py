#!/usr/bin/env python3
"""Run one exact frozen launcher after verifying the preinstalled runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    observed = {
        key: importlib.metadata.version(key).split("+", 1)[0] for key in expected
    }
    if observed != expected:
        raise RuntimeError(f"preinstalled frozen runtime mismatch: {observed}")
    if torch.__version__.split("+", 1)[0] != "2.5.1" or str(torch.version.cuda) != "12.1":
        raise RuntimeError(f"PyTorch/CUDA mismatch: {torch.__version__}, {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one and only one visible CUDA GPU is required")
    name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in name:
        raise RuntimeError(f"unexpected GPU: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--temp", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identities = {item["job_id"]: item for item in plan["jobs"]}
    first_line = args.launcher.read_bytes().split(b"\n", 1)[0]
    prefix = b"# HELDOUT_JOB_CONFIG_HEX="
    if not first_line.startswith(prefix):
        raise RuntimeError("launcher has no embedded job identity")
    embedded = json.loads(bytes.fromhex(first_line[len(prefix):].decode("ascii")))
    job_id = embedded["job_id"]
    identity = identities.get(job_id)
    if identity is None:
        raise RuntimeError(f"launcher not authorized by replacement plan: {job_id}")
    if args.launcher.stat().st_size != identity["launcher_bytes"] or sha256_file(args.launcher) != identity["launcher_sha256"]:
        raise RuntimeError(f"frozen launcher identity mismatch: {job_id}")
    verify_runtime()
    os.environ.update(
        KAGGLE_INPUT_PATH=str(args.input),
        KAGGLE_WORKING_PATH=str(args.working),
        KAGGLE_TEMP_PATH=str(args.temp),
    )
    spec = importlib.util.spec_from_file_location(f"frozen_{job_id.replace('-', '_')}", args.launcher)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_runtime = verify_runtime
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
