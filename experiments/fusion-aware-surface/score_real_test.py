#!/usr/bin/env python3
"""Score the sealed Scroll-4/5 cache only at preregistered thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cache_real_predictions import RUNS, sha256_file
from freeze_thresholds import score, validate_cache


METRICS = ("blend", "toposcore", "surface_dice", "voi_score")
SENSITIVITY_THRESHOLDS = (0.4, 0.5, 0.6)
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPLICATES = 10_000


def canonical_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def validate_embedded_hash(payload: dict, field: str = "payload_sha256") -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise RuntimeError(f"missing or invalid {field}")
    content = dict(payload)
    del content[field]
    if canonical_sha256(content) != observed:
        raise RuntimeError(f"{field} does not match payload")


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite vector of length >=2")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "patches": int(len(values)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def expected_test_files(split: dict) -> list[str]:
    records = [record for record in split["records"] if record["split"] == "test"]
    if len(records) < 20 or any(record["scroll"] not in {"s4", "s5"} for record in records):
        raise RuntimeError("test split is not the frozen Scroll-4/5 set")
    return sorted(Path(record["image"]).with_suffix(".npz").name for record in records)


def validate_run_cache(
    root: Path,
    run: str,
    expected: list[str],
    split_sha256: str,
    threshold: float,
) -> list[Path]:
    manifest_path = root / f"cache_manifest_{run}_test.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_embedded_hash(manifest)
    if manifest.get("status") != "test probabilities cached" or manifest.get("run") != run:
        raise RuntimeError(f"{run}: wrong test cache identity")
    if manifest.get("source_split_records_sha256") != split_sha256:
        raise RuntimeError(f"{run}: cache belongs to another split")
    if float(manifest.get("selected_threshold")) != threshold:
        raise RuntimeError(f"{run}: cache threshold disagrees with frozen threshold")
    files = {record["file"]: record for record in manifest.get("files", [])}
    if set(files) != set(expected):
        raise RuntimeError(f"{run}: cache manifest file set mismatch")
    directory = root / run
    observed = {path.name for path in directory.glob("*.npz")}
    if observed != set(expected):
        raise RuntimeError(f"{run}: cache directory file set mismatch")
    paths = []
    for name in expected:
        path = directory / name
        validate_cache(path)
        record = files[name]
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"{run}: cached file hash/size mismatch for {name}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--threshold-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--worker", type=Path, default=Path(__file__).with_name("official_metric.py")
    )
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    thresholds = json.loads(args.threshold_manifest.read_text(encoding="utf-8"))
    validate_embedded_hash(thresholds)
    if thresholds.get("status") != "thresholds frozen from Scroll-1 validation before test inference":
        raise RuntimeError("threshold manifest has the wrong status")
    if thresholds.get("source_split_records_sha256") != split.get("records_sha256"):
        raise RuntimeError("threshold and split manifests disagree")
    expected = expected_test_files(split)

    per_run: dict[str, dict] = {}
    selected_rows: dict[str, dict[str, dict[str, float]]] = {}
    for run in RUNS:
        selected = float(thresholds["runs"][run]["selected_threshold"])
        caches = validate_run_cache(
            args.test_root, run, expected, split["records_sha256"], selected
        )
        tested = sorted(set(SENSITIVITY_THRESHOLDS + (selected,)))
        threshold_records = {}
        for threshold in tested:
            rows = {
                cache.name: score(args.worker, cache, threshold) for cache in caches
            }
            threshold_records[str(threshold)] = {
                "means": {
                    metric: float(np.mean([row[metric] for row in rows.values()]))
                    for metric in METRICS
                },
                "per_patch": rows,
            }
            print(
                f"TEST run={run} threshold={threshold:.2f} "
                f"blend={threshold_records[str(threshold)]['means']['blend']:.6f}",
                flush=True,
            )
        selected_rows[run] = threshold_records[str(selected)]["per_patch"]
        per_run[run] = {
            "selected_threshold": selected,
            "thresholds": threshold_records,
        }

    paired = {}
    pooled_by_metric: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    for seed in (11, 23, 47):
        control = selected_rows[f"control_seed{seed}"]
        gap = selected_rows[f"gap8_seed{seed}"]
        if set(control) != set(gap):
            raise RuntimeError(f"seed {seed}: paired test patch mismatch")
        seed_record = {}
        for metric in METRICS:
            delta = np.asarray(
                [gap[name][metric] - control[name][metric] for name in sorted(control)]
            )
            seed_record[metric] = bootstrap_mean_interval(delta, seed=BOOTSTRAP_SEED + seed)
            pooled_by_metric[metric].append(delta)
        paired[str(seed)] = seed_record

    pooled = {}
    for metric in METRICS:
        per_patch_seed_mean = np.stack(pooled_by_metric[metric], axis=0).mean(axis=0)
        pooled[metric] = bootstrap_mean_interval(per_patch_seed_mean)
    gates = {
        "real_blend_noninferiority": pooled["blend"]["mean"] >= -0.005,
        "real_toposcore_improvement": pooled["toposcore"]["mean"] > 0.0,
    }
    payload = {
        "schema_version": "1.0",
        "status": "sealed Scroll-4/5 test scored at frozen thresholds",
        "source_split_records_sha256": split["records_sha256"],
        "source_threshold_payload_sha256": thresholds["payload_sha256"],
        "test_patch_count": len(expected),
        "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
        "bootstrap": {
            "unit": "held-out real patch",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95% descriptive interval",
        },
        "runs": per_run,
        "paired_gap8_minus_control": paired,
        "pooled_seed_mean_gap8_minus_control": pooled,
        "preregistered_real_gates": gates,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("real gates:", gates)
    print("payload SHA-256:", payload["payload_sha256"])
    print("SEALED_REAL_TEST_SCORED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
