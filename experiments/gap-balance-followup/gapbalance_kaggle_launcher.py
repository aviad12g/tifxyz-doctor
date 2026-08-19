#!/usr/bin/env python3
"""Fail-closed launcher for exactly one public GapBalance Kaggle training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path, PurePosixPath


ASSET_DATASET_ID = "aviadcohen1/vesuvius-fusion-aware-training-assets"
PUBLIC_SOURCE_COMMIT = "0bdca375c56fc6b32dfc3cccbdd1226206bac0b2"
PUBLIC_SOURCE_BASE = (
    "https://raw.githubusercontent.com/aviad12g/tifxyz-doctor/"
    f"{PUBLIC_SOURCE_COMMIT}/experiments/fusion-aware-surface"
)
PUBLIC_REPLACEMENTS = {
    "project/kaggle_train_runner.py": "6aa86b362aac0575f8cb4e7896a49e865576e7599317a07070cd6aab14621d12",
    "project/train_fusion_aware.py": "d427dfcfb5f4946d366ba86a6c9725d8e1caf7264691549a29db65225b186a97",
}
FIXED_ASSET_PATHS = (
    "archives/images_s1.tar",
    "archives/images_s4_s5.tar",
    "archives/labels.tar",
    "model/Model_epoch499.pth",
    "painter/contrast_phantom.py",
    "painter/synthetic_scroll_twin.py",
    "diagnostic/loader059.py",
    "diagnostic/fusion_readout.py",
)
SOURCE_LEDGER_SHA256 = "1b3d78b2f85808a4a2953b7b8ed3a5969f07714f341cea269fb11746f7892fba"
ALLOWED_JOBS = {
    ("gap2", 11),
    ("gap2", 23),
    ("gap2", 47),
    ("gap4", 11),
    ("gap4", 23),
    ("gap4", 47),
}
FROZEN_ARM = "__FROZEN_ARM__"
FROZEN_SEED = "__FROZEN_SEED__"
FROZEN_VERIFY_ONLY = "__FROZEN_VERIFY_ONLY__"


class LaunchError(RuntimeError):
    """The exact public training projection could not be established."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


def resolve_job(requested_arm: str | None, requested_seed: int | None) -> tuple[str, int]:
    if FROZEN_ARM.startswith("__FROZEN_"):
        require(requested_arm is not None and requested_seed is not None, "arm and seed are required")
        job = (requested_arm, requested_seed)
    else:
        frozen_seed = int(FROZEN_SEED)
        require(requested_arm in {None, FROZEN_ARM}, "requested arm differs from frozen kernel")
        require(requested_seed in {None, frozen_seed}, "requested seed differs from frozen kernel")
        job = (FROZEN_ARM, frozen_seed)
    require(job in ALLOWED_JOBS, f"job is outside the frozen six-job plan: {job}")
    return job


def resolve_verify_only(requested: bool) -> bool:
    if isinstance(FROZEN_VERIFY_ONLY, str) and FROZEN_VERIFY_ONLY.startswith("__FROZEN_"):
        return requested
    return bool(FROZEN_VERIFY_ONLY)


def find_asset_root(search_root: Path) -> Path:
    matches = []
    for metadata in search_root.glob("**/dataset-metadata.json"):
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("id") == ASSET_DATASET_ID:
            matches.append(metadata.parent.resolve())
    require(len(matches) == 1, f"expected one mounted {ASSET_DATASET_ID}; found {matches}")
    return matches[0]


def load_public_file(relative: str, local_source_root: Path | None) -> bytes:
    public_relative = relative.removeprefix("project/")
    if local_source_root is not None:
        raw = (local_source_root / public_relative).read_bytes()
    else:
        url = f"{PUBLIC_SOURCE_BASE}/{public_relative}"
        request = urllib.request.Request(url, headers={"User-Agent": "gapbalance-frozen-launcher/1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            require(response.status == 200, f"public source returned HTTP {response.status}")
            raw = response.read()
    require(sha256_bytes(raw) == PUBLIC_REPLACEMENTS[relative], f"public hash mismatch: {relative}")
    return raw


def read_source_ledger(asset_root: Path) -> list[tuple[str, str]]:
    ledger = asset_root / "SOURCE_SHA256SUMS"
    require(ledger.is_file(), "source ledger is missing")
    require(sha256_file(ledger) == SOURCE_LEDGER_SHA256, "source ledger identity changed")
    records = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise LaunchError(f"invalid source-ledger line {line_number}") from error
        path = PurePosixPath(relative)
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"invalid source-ledger hash at line {line_number}",
        )
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe ledger path: {relative}")
        require((asset_root / relative).is_file(), f"missing source-ledger file: {relative}")
        records.append((digest, relative))
    require(len(records) >= 200, "source ledger is unexpectedly short")
    return records


def build_projection(
    asset_root: Path,
    projection: Path,
    local_source_root: Path | None,
) -> dict[str, str]:
    require(not projection.exists(), f"projection must start absent: {projection}")
    records = read_source_ledger(asset_root)
    replacements = {
        relative: load_public_file(relative, local_source_root)
        for relative in PUBLIC_REPLACEMENTS
    }
    projected_ledger = []
    for original_digest, relative in records:
        target = projection / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in replacements:
            raw = replacements[relative]
            target.write_bytes(raw)
            digest = sha256_bytes(raw)
        else:
            target.symlink_to((asset_root / relative).resolve())
            digest = original_digest
        projected_ledger.append((digest, relative))
    for relative in FIXED_ASSET_PATHS:
        source = asset_root / relative
        require(source.is_file(), f"missing fixed asset: {relative}")
        target = projection / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
    ledger = projection / "SOURCE_SHA256SUMS"
    ledger.write_text(
        "".join(f"{digest}  {relative}\n" for digest, relative in projected_ledger),
        encoding="utf-8",
    )
    return {
        "original_source_ledger_sha256": SOURCE_LEDGER_SHA256,
        "projected_source_ledger_sha256": sha256_file(ledger),
        **{relative: digest for relative, digest in PUBLIC_REPLACEMENTS.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("gap2", "gap4"))
    parser.add_argument("--seed", choices=(11, 23, 47), type=int)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--public-source-root", type=Path)
    parser.add_argument("--projection", type=Path, default=Path("/kaggle/working/gapbalance-projection"))
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working/gapbalance"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    arm, seed = resolve_job(args.arm, args.seed)
    verify_only = resolve_verify_only(args.verify_only)
    asset_root = (
        args.asset_root.resolve()
        if args.asset_root is not None
        else find_asset_root(Path(os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input")))
    )
    local_source_root = args.public_source_root.resolve() if args.public_source_root else None
    identities = build_projection(asset_root, args.projection.resolve(), local_source_root)
    print(json.dumps({
        "status": "GAPBALANCE_PUBLIC_PROJECTION_READY",
        "arm": arm,
        "seed": seed,
        "public_source_commit": PUBLIC_SOURCE_COMMIT,
        "identities": identities,
        "confirmation_outputs_inspected": False,
    }, sort_keys=True), flush=True)

    runner = args.projection.resolve() / "project" / "kaggle_train_runner.py"
    command = [
        sys.executable,
        str(runner),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--asset-root",
        str(args.projection.resolve()),
        "--work",
        str(args.work.resolve()),
    ]
    if verify_only:
        command.append("--verify-only")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
