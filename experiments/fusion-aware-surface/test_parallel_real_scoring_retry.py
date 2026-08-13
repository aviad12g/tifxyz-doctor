#!/usr/bin/env python3
"""Result-blind tests for the accelerated real-scoring retry."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import score_real_test as scorer


def test_parallel_scoring_preserves_frozen_cache_order(monkeypatch) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def fake_score(worker: Path, cache: Path, threshold: float) -> dict[str, float]:
        del worker
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02 * (5 - int(cache.stem)))
        with lock:
            active -= 1
        value = int(cache.stem) + threshold
        return {
            "blend": value,
            "toposcore": value + 1,
            "surface_dice": value + 2,
            "voi_score": value + 3,
        }

    monkeypatch.setattr(scorer, "score", fake_score)
    caches = [Path(f"{index}.npz") for index in range(1, 5)]
    sequential = scorer.score_cache_set(Path("worker.py"), caches, 0.5, 1)
    parallel = scorer.score_cache_set(Path("worker.py"), caches, 0.5, 4)
    assert parallel == sequential
    assert list(parallel) == [path.name for path in caches]
    assert maximum_active > 1


def test_parallel_scoring_rejects_unfrozen_worker_counts() -> None:
    with pytest.raises(ValueError, match="parallel workers"):
        scorer.score_cache_set(Path("worker.py"), [], 0.5, 5)


def test_launcher_runs_panel_fail_fast_before_long_real_scorer() -> None:
    source = Path(__file__).with_name("one_shot_scoring_launcher.py").read_text(
        encoding="utf-8"
    )
    main = source[source.index("def main() -> int:") :]
    panel = main.index("panel_invocation = execute_real_panels(")
    scoring = main.index("result_path, invocation = execute_scorer(")
    assert panel < scoring
    assert '"--parallel-workers",\n                "4"' in source
    assert "sys.stderr.buffer.write(completed.stderr)" in source
