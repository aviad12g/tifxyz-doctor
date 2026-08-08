#!/usr/bin/env python3
"""Subprocess-isolated official Vesuvius leaderboard metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


METRIC_DATASET = "sohier/vesuvius-metric-resources"
METRIC_ZIP_SHA256 = "64d24044b7381dbb660a0e9f602ad0f8fd37d1935c09a8ee3b78088433910e68"
METRIC_SOURCE_HASHES = {
    "src/topometrics/__init__.py": "9f1f546d4404e10619fd2da78de6b3d7dbf91a078d4de764476bf21b7c37b69e",
    "src/topometrics/_bm_loader.py": "c2abf08c60341f8da685c14d89a15f9f4bcfa4bc331139e3c78d5351e8a679eb",
    "src/topometrics/leaderboard.py": "f0db94436eea4464a30f252ebc7c35553e539da5e4832e4efaf523ed664cd811",
    "src/topometrics/toposcore.py": "ba09b873d40f7bc8681fcab9c0d5260d117ceb79bc2870f6d65cd18a4c5017c0",
    "src/topometrics/voi.py": "1c8e5131c6102baa2b5e01539b6dde813a35bfc737d157c759c76d598732dc47",
    "external/Betti-Matching-3D/CMakeLists.txt": "1dd442109ca75cd64b2399acab82721e4b7d3d9b8240b7b7afe289bf5fdc81a2",
    "external/Betti-Matching-3D/src/_BettiMatching.cpp": "7b8c41dc4f9154abfdeaa392ae8e928f20f1967f3cb4405c7ef1b159a227fe2e",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_metric_root() -> Path:
    configured = os.environ.get("FUSION_TOPOMETRICS_ROOT")
    if configured:
        return Path(configured).resolve()
    return (
        Path(__file__).resolve().parent.parent
        / "metric-resources"
        / "topological-metrics-kaggle"
    )


def verify_metric_source(root: Path) -> None:
    for relative, expected in METRIC_SOURCE_HASHES.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"official metric source mismatch: {relative}")
    build = root / "external" / "Betti-Matching-3D" / "build"
    if not any(build.glob("betti_matching*.so")):
        raise RuntimeError("official Betti-Matching module has not been built")


def load_leaderboard():
    root = resolve_metric_root()
    verify_metric_source(root)
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import topometrics.leaderboard as leaderboard

    return leaderboard


def compute_official(gt: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    leaderboard = load_leaderboard()

    result = leaderboard.compute_leaderboard_score(
        predictions=prediction.astype(np.uint8),
        labels=gt.astype(np.uint8),
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=2.0,
        voi_connectivity=26,
        voi_transform="one_over_one_plus",
        voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35),
        fg_threshold=None,
        ignore_label=2,
        ignore_mask=None,
    )
    return {
        "blend": float(result.score),
        "toposcore": float(result.topo.toposcore),
        "surface_dice": float(result.surface_dice),
        "voi_score": float(result.voi.voi_score),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("threshold", type=float)
    args = parser.parse_args()
    with np.load(args.cache) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"{args.cache}: expected exactly prob and gt")
        probability = np.asarray(data["prob"], dtype=np.float32)
        gt = np.asarray(data["gt"]) > 0
    if probability.shape != gt.shape:
        raise RuntimeError(f"{args.cache}: probability/GT shape mismatch")
    print(json.dumps(compute_official(gt, probability >= args.threshold), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
