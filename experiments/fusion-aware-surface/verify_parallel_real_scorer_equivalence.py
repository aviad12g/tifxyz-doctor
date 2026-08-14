#!/usr/bin/env python3
"""Prove concurrent cache scoring equals sequential scoring on public fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import score_real_test_parallel as parallel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_public_fixtures(root: Path) -> list[Path]:
    coordinates = np.indices((8, 8, 8), dtype=np.float32)
    fixtures = []
    for index in range(4):
        distance = np.sqrt(
            (coordinates[0] - (3.0 + 0.2 * index)) ** 2
            + (coordinates[1] - (3.5 - 0.1 * index)) ** 2
            + (coordinates[2] - (3.0 + 0.1 * index)) ** 2
        )
        gt = (distance <= (2.4 + 0.1 * index)).astype(np.uint8)
        probability = np.clip(
            1.0 - distance / (4.8 + 0.2 * index), 0.0, 1.0
        ).astype(np.float16)
        path = root / f"public_fixture_{index}.npz"
        np.savez_compressed(path, prob=probability, gt=gt)
        fixtures.append(path)
    return fixtures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--exact-scorer", type=Path, required=True)
    parser.add_argument("--parallel-scorer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("equivalence-report target must start absent")
    os.environ["FUSION_TOPOMETRICS_ROOT"] = str(args.metric_root.resolve())
    with tempfile.TemporaryDirectory(prefix="parallel-real-equivalence-") as temporary:
        fixtures = build_public_fixtures(Path(temporary))
        for fixture in fixtures:
            parallel.validate_cache(fixture)
        sequential = parallel.score_cache_set(args.worker, fixtures, 0.5, 1)
        sequential_sha256 = canonical_sha256(sequential)
        comparisons = []
        for workers in (4, 16, 32):
            observed = parallel.score_cache_set(args.worker, fixtures, 0.5, workers)
            comparisons.append(
                {
                    "parallel_workers": workers,
                    "result_sha256": canonical_sha256(observed),
                    "exact_python_value_equality": observed == sequential,
                    "frozen_cache_order_preserved": list(observed) == [
                        fixture.name for fixture in fixtures
                    ],
                }
            )
    payload = {
        "schema_version": "1.0",
        "status": "public official-metric parallel scorer equivalence verified",
        "fixture_scope": {
            "held_out_cache_used": False,
            "synthetic_public_fixture_count": 4,
            "shape_zyx": [8, 8, 8],
            "threshold": 0.5,
        },
        "exact_scorer": {
            "file": args.exact_scorer.name,
            "sha256": sha256_file(args.exact_scorer),
        },
        "parallel_scorer": {
            "file": args.parallel_scorer.name,
            "sha256": sha256_file(args.parallel_scorer),
        },
        "official_metric_worker": {
            "file": args.worker.name,
            "sha256": sha256_file(args.worker),
        },
        "sequential_result_sha256": sequential_sha256,
        "comparisons": comparisons,
        "scientific_contract": {
            "per_cache_metric_subprocess_changed": False,
            "cache_threshold_or_input_changed": False,
            "aggregation_bootstrap_or_serialization_changed": False,
            "only_independent_subprocess_scheduling_changed": True,
        },
    }
    if not all(
        record["exact_python_value_equality"]
        and record["frozen_cache_order_preserved"]
        and record["result_sha256"] == sequential_sha256
        for record in comparisons
    ):
        raise RuntimeError("parallel official-metric result differs from sequential result")
    payload["payload_sha256"] = canonical_sha256(payload)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PUBLIC_PARALLEL_REAL_SCORER_EQUIVALENCE_VERIFIED")
    print(payload["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
