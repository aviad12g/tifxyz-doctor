#!/usr/bin/env python3
"""Select and hash real-validation thresholds before test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


THRESHOLDS = tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_threshold(mean_blend: dict[float, float]) -> float:
    """Maximize mean blend; tie by distance to 0.5, then lower threshold."""
    return min(
        mean_blend,
        key=lambda threshold: (
            -mean_blend[threshold],
            abs(threshold - 0.5),
            threshold,
        ),
    )


def validate_cache(path: Path) -> None:
    with np.load(path) as data:
        if set(data.files) != {"prob", "gt"}:
            raise RuntimeError(f"{path}: expected exactly prob and gt")
        probability = np.asarray(data["prob"])
        gt = np.asarray(data["gt"])
        if probability.shape != gt.shape or probability.ndim != 3:
            raise RuntimeError(f"{path}: invalid paired ZYX shapes")
        if probability.dtype != np.float16:
            raise RuntimeError(f"{path}: probability must be float16")
        if gt.dtype != np.uint8 or np.any((gt != 0) & (gt != 1)):
            raise RuntimeError(f"{path}: GT must be binary uint8")
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise RuntimeError(f"{path}: invalid probability values")


def score(worker: Path, cache: Path, threshold: float) -> dict[str, float]:
    result = subprocess.run(
        [sys.executable, str(worker), str(cache), str(threshold)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"official metric failed for {cache.name}@{threshold}: "
            f"{result.stderr.strip()}"
        )
    values = json.loads(result.stdout)
    if set(values) != {"blend", "toposcore", "surface_dice", "voi_score"}:
        raise RuntimeError("official metric returned an unexpected schema")
    if not all(np.isfinite(float(value)) for value in values.values()):
        raise RuntimeError("official metric returned a non-finite value")
    return {key: float(value) for key, value in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worker",
        type=Path,
        default=Path(__file__).with_name("official_metric.py"),
    )
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validation = sorted(
        record["image"]
        for record in split["records"]
        if record["split"] == "validation"
    )
    if len(validation) < 20:
        raise RuntimeError(f"insufficient validation patches: {len(validation)}")
    expected_files = {Path(name).with_suffix(".npz").name for name in validation}

    payload = {
        "schema_version": "1.0",
        "status": "thresholds frozen from Scroll-1 validation before test inference",
        "candidate_thresholds": list(THRESHOLDS),
        "tie_break": "maximum mean official blend; nearest 0.5; lower threshold",
        "source_split_records_sha256": split["records_sha256"],
        "validation_patch_count": len(validation),
        "runs": {},
    }
    for run in RUNS:
        directory = args.validation_root / run
        observed = {path.name for path in directory.glob("*.npz")}
        if observed != expected_files:
            raise RuntimeError(
                f"{run}: validation cache mismatch; "
                f"missing={sorted(expected_files-observed)} extra={sorted(observed-expected_files)}"
            )
        caches = [directory / name for name in sorted(expected_files)]
        for cache in caches:
            validate_cache(cache)
        rows = {}
        mean_blend = {}
        for threshold in THRESHOLDS:
            threshold_rows = [score(args.worker, cache, threshold) for cache in caches]
            rows[str(threshold)] = threshold_rows
            mean_blend[threshold] = float(
                np.mean([record["blend"] for record in threshold_rows])
            )
            print(
                f"THRESHOLD run={run} threshold={threshold:.2f} "
                f"mean_blend={mean_blend[threshold]:.6f}",
                flush=True,
            )
        selected = choose_threshold(mean_blend)
        payload["runs"][run] = {
            "selected_threshold": selected,
            "mean_blend": {str(key): value for key, value in mean_blend.items()},
            "per_patch_official": rows,
            "validation_files": [
                {
                    "file": cache.name,
                    "sha256": sha256_file(cache),
                    "bytes": cache.stat().st_size,
                }
                for cache in caches
            ],
        }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "selected:",
        {run: record["selected_threshold"] for run, record in payload["runs"].items()},
    )
    print("payload SHA-256:", payload["payload_sha256"])
    print("VALIDATION_THRESHOLDS_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
