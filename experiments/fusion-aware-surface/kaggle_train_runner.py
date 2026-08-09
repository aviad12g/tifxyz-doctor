#!/usr/bin/env python3
"""Fail-closed Kaggle launcher for one frozen fusion-aware training arm/seed.

This script verifies the private input bundle, installs the preregistered
runtime, extracts the public Dataset059 archives, launches exactly one of the
six training jobs, and emits a content-addressed run manifest. It performs no
validation or test inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path


DATASET_ID = "aviadcohen1/vesuvius-fusion-aware-training-assets"
CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
ASSETS = {
    "archives/images_s1.tar": (
        1_478_256_640,
        "cce01d96c77cc7966a41a805b2690db3e7705ce92cbc52d35d0a83a5c7ba35a5",
    ),
    "archives/images_s4_s5.tar": (
        444_723_200,
        "960a152238df8fc60d108a13302af9676c096019c21f204447aa3bf7d9dce4ae",
    ),
    "archives/labels.tar": (
        83_087_360,
        "6e01e4d5f0591796a060bda0a1357ed8cc8b801912a7d3e569ecb246764741a6",
    ),
    "model/Model_epoch499.pth": (819_171_665, CHECKPOINT_SHA256),
    "painter/contrast_phantom.py": (
        None,
        "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8",
    ),
    "painter/synthetic_scroll_twin.py": (
        None,
        "9cc3b24ef34c75118e5e348ef472a9e3354414997022550bcb2aab9921578710",
    ),
    "diagnostic/loader059.py": (
        None,
        "49a1d4ea3ee611236d53b1291ca2e8cd6e1450f2fc032223ff89a5b53d9ec903",
    ),
    "diagnostic/fusion_readout.py": (
        None,
        "a533ee940712f1a47111705e4d8fff4990ccffb0afb103c7337a68bff244781c",
    ),
}
REQUIRED_PROJECT_FILES = (
    "PREREGISTRATION.md",
    "real_split_manifest.json",
    "train_fusion_aware.py",
    "fusion_loss.py",
    "gap_supervision.py",
    "normalization.py",
)
PIP_PACKAGES = (
    "numpy==1.26.4",
    "tifffile==2025.2.18",
    "imagecodecs==2024.12.30",
    "huggingface-hub==1.11.0",
    "timm==1.0.15",
    "einops==0.8.1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_asset_root(search_root: Path) -> Path:
    matches = []
    for checkpoint in search_root.glob("**/model/Model_epoch499.pth"):
        root = checkpoint.parent.parent
        metadata = root / "dataset-metadata.json"
        if not metadata.exists():
            continue
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if payload.get("id") == DATASET_ID:
            matches.append(root)
    if len(matches) != 1:
        raise RuntimeError(f"expected one mounted {DATASET_ID} bundle; found {matches}")
    return matches[0]


def verify_fixed_assets(root: Path) -> dict[str, str]:
    verified = {}
    for relative, (expected_size, expected_hash) in ASSETS.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing frozen asset: {relative}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch for {relative}: {path.stat().st_size}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: {actual}")
        verified[relative] = actual
    return verified


def verify_source_ledger(root: Path) -> tuple[str, int]:
    ledger = root / "SOURCE_SHA256SUMS"
    if not ledger.is_file():
        raise RuntimeError("missing SOURCE_SHA256SUMS")
    count = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source ledger mismatch: {relative}")
        count += 1
    if count < 200:
        raise RuntimeError(f"source ledger unexpectedly short: {count}")
    return sha256_file(ledger), count


def verify_project(root: Path) -> dict[str, str]:
    project = root / "project"
    hashes = {}
    for relative in REQUIRED_PROJECT_FILES:
        path = project / relative
        if not path.is_file():
            raise RuntimeError(f"missing project file: {relative}")
        hashes[relative] = sha256_file(path)
    split = json.loads((project / "real_split_manifest.json").read_text(encoding="utf-8"))
    expected_counts = {
        "s1:train": 138,
        "s1:validation": 24,
        "s4:test": 36,
        "s5:test": 2,
    }
    if split.get("counts") != expected_counts:
        raise RuntimeError(f"unexpected frozen split counts: {split.get('counts')}")
    return hashes


def install_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "torch==2.5.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *PIP_PACKAGES],
        check=True,
    )


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r") as bundle:
        for member in bundle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive link is not permitted: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"archive path traversal: {member.name}")
        bundle.extractall(destination)


def extract_training_data(root: Path, destination: Path) -> None:
    for name in ("images_s1.tar", "labels.tar"):
        marker = destination / f".{name}.complete"
        if marker.exists():
            continue
        safe_extract(root / "archives" / name, destination)
        marker.write_text(ASSETS[f"archives/{name}"][1] + "\n", encoding="utf-8")
    image_count = len(list((destination / "imagesTr").glob("s1_*_0000.tif")))
    label_count = len(list((destination / "labelsTr").glob("s1_*.tif")))
    if (image_count, label_count) != (162, 162):
        raise RuntimeError(f"unexpected Scroll-1 extraction: images={image_count} labels={label_count}")


def stream_command(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "gap8"), required=True)
    parser.add_argument("--seed", choices=(11, 23, 47), type=int, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working/fusion-aware"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    search_root = Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    root = args.asset_root.resolve() if args.asset_root else find_asset_root(search_root)
    fixed_hashes = verify_fixed_assets(root)
    ledger_hash, ledger_count = verify_source_ledger(root)
    project_hashes = verify_project(root)
    print(
        f"FROZEN_ASSETS_VERIFIED fixed={len(fixed_hashes)} source_files={ledger_count} "
        f"ledger_sha256={ledger_hash}",
        flush=True,
    )
    if args.verify_only:
        return 0

    install_runtime()
    import torch

    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"unexpected torch runtime: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    work = args.work.resolve()
    real_data = work / "real-data"
    run_work = work / f"{args.arm}-seed{args.seed}"
    run_work.mkdir(parents=True, exist_ok=True)
    extract_training_data(root, real_data)

    network_src = root / "network-source" / "vesuvius" / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(network_src) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(root / "project" / "train_fusion_aware.py"),
        "--arm",
        args.arm,
        "--seed",
        str(args.seed),
        "--real-data",
        str(real_data),
        "--split-manifest",
        str(root / "project" / "real_split_manifest.json"),
        "--painter-scripts",
        str(root / "painter"),
        "--diagnostic-scripts",
        str(root / "diagnostic"),
        "--checkpoint",
        str(root / "model" / "Model_epoch499.pth"),
        "--work",
        str(run_work),
    ]
    log = run_work / "training.log"
    stream_command(command, cwd=root / "project", env=env, log=log)

    stem = f"surface059_{args.arm}_seed{args.seed}"
    checkpoint = run_work / "checkpoints" / f"{stem}.pth"
    metadata_path = run_work / "checkpoints" / f"{stem}.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise RuntimeError("training completed without the expected checkpoint pair")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "arm": args.arm,
        "seed": args.seed,
        "steps": 1500,
        "gap_weight": 1.0 if args.arm == "control" else 8.0,
        "real_train_count": 138,
        "real_validation_count": 24,
        "synthetic_count": 16,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "normalization_scheme": "CTNormalization",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"output metadata mismatch for {key}: {metadata.get(key)!r}")
    if sha256_file(checkpoint) != metadata.get("checkpoint_sha256"):
        raise RuntimeError("output checkpoint hash does not match its metadata")

    manifest = {
        "schema_version": "1.0",
        "status": "training complete; validation and test endpoints not computed",
        "arm": args.arm,
        "seed": args.seed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "numpy": package_version("numpy"),
            "tifffile": package_version("tifffile"),
            "imagecodecs": package_version("imagecodecs"),
            "timm": package_version("timm"),
            "einops": package_version("einops"),
        },
        "inputs": {
            "dataset_id": DATASET_ID,
            "fixed_assets": fixed_hashes,
            "source_ledger_sha256": ledger_hash,
            "source_ledger_count": ledger_count,
            "project_files": project_hashes,
        },
        "outputs": {
            "checkpoint": str(checkpoint.relative_to(work)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metadata": str(metadata_path.relative_to(work)),
            "metadata_sha256": sha256_file(metadata_path),
            "training_log": str(log.relative_to(work)),
            "training_log_sha256": sha256_file(log),
        },
    }
    manifest_path = run_work / "training_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"TRAINING_RUN_VERIFIED manifest={manifest_path} sha256={sha256_file(manifest_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
