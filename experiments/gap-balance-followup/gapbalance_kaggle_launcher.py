#!/usr/bin/env python3
"""Fail-closed launcher for exactly one public GapBalance Kaggle training job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
ARCHIVE_NAMES = ("images_s1", "images_s4_s5", "labels")
EXPANDED_ARCHIVE_COUNT = 400
EXPANDED_ARCHIVE_BYTES = 2_005_739_905
EXPANDED_ARCHIVE_SHA256 = "981448a526d2e04cb59b0b1f40331fa901ece5c0514161733814c5b7ea823021"
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
    for ledger in search_root.glob("**/SOURCE_SHA256SUMS"):
        try:
            if sha256_file(ledger) != SOURCE_LEDGER_SHA256:
                continue
        except OSError:
            continue
        matches.append(ledger.parent.resolve())
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


def expanded_archive_identity(root: Path) -> tuple[int, int, str]:
    rows = []
    total_bytes = 0
    for archive_name in ARCHIVE_NAMES:
        archive_root = root / "archives" / archive_name
        require(archive_root.is_dir(), f"expanded archive directory missing: {archive_root}")
        for path in sorted(archive_root.rglob("*")):
            if not path.is_file():
                continue
            size = path.stat().st_size
            relative = path.relative_to(archive_root).as_posix()
            rows.append(f"{sha256_file(path)} {size} {archive_name}/{relative}\n")
            total_bytes += size
    payload = "".join(sorted(rows)).encode("utf-8")
    return len(rows), total_bytes, sha256_bytes(payload)


def verify_fixed_assets_compat(frozen, root: Path, original_verify) -> dict[str, str]:
    tar_paths = [root / "archives" / f"{name}.tar" for name in ARCHIVE_NAMES]
    if all(path.is_file() for path in tar_paths):
        return original_verify(root)
    verified = {}
    for relative, (expected_size, expected_hash) in frozen.ASSETS.items():
        if relative.startswith("archives/"):
            continue
        path = root / relative
        require(path.is_file(), f"fixed asset missing: {relative}")
        require(expected_size is None or path.stat().st_size == expected_size, f"fixed asset wrong-sized: {relative}")
        actual_hash = sha256_file(path)
        require(actual_hash == expected_hash, f"fixed asset hash mismatch: {relative}")
        verified[relative] = actual_hash
    identity = expanded_archive_identity(root)
    expected = (EXPANDED_ARCHIVE_COUNT, EXPANDED_ARCHIVE_BYTES, EXPANDED_ARCHIVE_SHA256)
    require(identity == expected, f"expanded archive identity mismatch: {identity} != {expected}")
    verified["archives/expanded-content"] = identity[2]
    return verified


def extract_training_data_compat(root: Path, destination: Path, original_extract) -> None:
    tar_paths = [root / "archives" / f"{name}.tar" for name in ARCHIVE_NAMES]
    if all(path.is_file() for path in tar_paths):
        original_extract(root, destination)
        return
    for archive_name in ("images_s1", "labels"):
        archive_root = root / "archives" / archive_name
        for source in sorted(archive_root.rglob("*")):
            if not source.is_file():
                continue
            target = destination / source.relative_to(archive_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), f"duplicate expanded training asset: {target}")
            target.symlink_to(source)
    image_count = len(list((destination / "imagesTr").glob("s1_*_0000.tif")))
    label_count = len(list((destination / "labelsTr").glob("s1_*.tif")))
    require((image_count, label_count) == (162, 162), f"unexpected Scroll-1 expansion: images={image_count} labels={label_count}")


def configure_torch_compiler_compat(torch) -> None:
    import inspect

    if "reason" not in inspect.signature(torch.compiler.disable).parameters:
        original_disable = torch.compiler.disable

        def disable_compat(fn=None, recursive=True, *, reason=None):
            return original_disable(fn=fn, recursive=recursive)

        torch.compiler.disable = disable_compat
    compat_root = Path("/kaggle/working/fusion-runtime-compat")
    compat_root.mkdir(parents=True, exist_ok=True)
    (compat_root / "sitecustomize.py").write_text(
        "import inspect\n"
        "import torch\n"
        "if 'reason' not in inspect.signature(torch.compiler.disable).parameters:\n"
        "    _original_disable = torch.compiler.disable\n"
        "    def _disable_compat(fn=None, recursive=True, *, reason=None):\n"
        "        return _original_disable(fn=fn, recursive=recursive)\n"
        "    torch.compiler.disable = _disable_compat\n",
        encoding="utf-8",
    )
    os.environ["PYTHONPATH"] = str(compat_root) + os.pathsep + os.environ.get("PYTHONPATH", "")


def run_model_loader_preflight(source_root: Path) -> None:
    network_src = source_root / "network-source" / "vesuvius" / "src"
    diagnostic_src = source_root / "diagnostic"
    sys.path.insert(0, str(network_src))
    sys.path.insert(0, str(diagnostic_src))
    import torch
    import loader059

    configure_torch_compiler_compat(torch)
    loader059.CKPT = str(source_root / "model" / "Model_epoch499.pth")
    model, _, _ = loader059.load_059()
    require(next(model.parameters()).is_cuda and not model.training, "full model-loader preflight failed")
    del model
    torch.cuda.empty_cache()
    print("MODEL_LOADER_PREFLIGHT_OK timm=1.0.27", flush=True)


def install_runtime_compat(frozen, source_root: Path) -> None:
    packages = tuple(package for package in frozen.PIP_PACKAGES if not package.startswith("timm==")) + (
        "timm==1.0.27",
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *packages],
        check=True,
    )
    subprocess.run(
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
        check=True,
    )
    run_model_loader_preflight(source_root)


def execute_projected_runner(runner: Path, source_root: Path, arm: str, seed: int, work: Path, verify_only: bool) -> int:
    spec = importlib.util.spec_from_file_location("gapbalance_projected_runner", runner)
    require(spec is not None and spec.loader is not None, f"cannot load projected runner: {runner}")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    original_verify = frozen.verify_fixed_assets
    original_extract = frozen.extract_training_data
    frozen.verify_fixed_assets = lambda root: verify_fixed_assets_compat(frozen, root, original_verify)
    frozen.extract_training_data = lambda root, destination: extract_training_data_compat(root, destination, original_extract)
    frozen.install_runtime = lambda: install_runtime_compat(frozen, source_root)

    original_argv = sys.argv
    sys.argv = [
        str(runner),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--asset-root",
        str(source_root),
        "--work",
        str(work),
    ]
    if verify_only:
        sys.argv.append("--verify-only")
    try:
        return int(frozen.main())
    finally:
        sys.argv = original_argv


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
        if not source.is_file() and relative.startswith("archives/") and relative.endswith(".tar"):
            expanded_source = asset_root / relative.removesuffix(".tar")
            require(expanded_source.is_dir(), f"missing fixed asset: {relative}")
            expanded_target = projection / relative.removesuffix(".tar")
            if not expanded_target.exists():
                expanded_target.parent.mkdir(parents=True, exist_ok=True)
                expanded_target.symlink_to(expanded_source.resolve())
            continue
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
    return execute_projected_runner(
        runner,
        args.projection.resolve(),
        arm,
        seed,
        args.work.resolve(),
        verify_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
