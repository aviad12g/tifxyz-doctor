"""Result-blind equivalence checks for concurrent real-cache scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import score_real_test as exact
import score_real_test_parallel as parallel


def _write_hashed(path: Path, payload: dict) -> None:
    content = dict(payload)
    content["payload_sha256"] = exact.canonical_sha256(content)
    path.write_text(json.dumps(content), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    split = {
        "records_sha256": "1" * 64,
        "records": [
            {
                "split": "test",
                "scroll": "s4" if index % 2 == 0 else "s5",
                "image": f"patch_{index:02d}.tif",
            }
            for index in range(20)
        ],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    thresholds = {
        "status": "thresholds frozen from Scroll-1 validation before test inference",
        "source_split_records_sha256": split["records_sha256"],
        "runs": {
            run: {"selected_threshold": 0.45}
            for run in exact.RUNS
        },
    }
    threshold_path = tmp_path / "thresholds.json"
    _write_hashed(threshold_path, thresholds)
    root = tmp_path / "caches"
    names = [f"patch_{index:02d}.npz" for index in range(20)]
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    for run in exact.RUNS:
        directory = root / run
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).touch()
        manifest = {
            "status": "test probabilities cached",
            "run": run,
            "source_split_records_sha256": split["records_sha256"],
            "selected_threshold": 0.45,
            "files": [
                {"file": name, "sha256": empty_sha256, "bytes": 0}
                for name in names
            ],
        }
        _write_hashed(root / f"cache_manifest_{run}_test.json", manifest)
    return root, split_path, threshold_path


def _fake_score(worker: Path, cache: Path, threshold: float) -> dict[str, float]:
    del worker
    run_index = exact.RUNS.index(cache.parent.name)
    patch_index = int(cache.stem.rsplit("_", 1)[1])
    base = run_index * 0.001 + patch_index * 0.00001 + threshold * 0.01
    return {
        "blend": base,
        "toposcore": base + 0.1,
        "surface_dice": base + 0.2,
        "voi_score": base + 0.3,
    }


def _run(module, monkeypatch, argv: list[str]) -> bytes:
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    return Path(argv[argv.index("--out") + 1]).read_bytes()


def test_parallel_main_is_byte_identical_to_exact_for_all_worker_shapes(
    tmp_path, monkeypatch
) -> None:
    root, split, thresholds = _fixture(tmp_path)
    monkeypatch.setattr(exact, "validate_cache", lambda path: None)
    monkeypatch.setattr(parallel, "validate_cache", lambda path: None)
    monkeypatch.setattr(exact, "score", _fake_score)
    monkeypatch.setattr(parallel, "score", _fake_score)
    exact_out = tmp_path / "exact.json"
    common = [
        "scorer",
        "--test-root",
        str(root),
        "--split-manifest",
        str(split),
        "--threshold-manifest",
        str(thresholds),
        "--out",
    ]
    expected = _run(exact, monkeypatch, common + [str(exact_out)])
    for workers in (1, 4, 16, 32):
        out = tmp_path / f"parallel-{workers}.json"
        observed = _run(
            parallel,
            monkeypatch,
            common + [str(out), "--parallel-workers", str(workers)],
        )
        assert observed == expected


def test_parallel_worker_completion_order_cannot_change_result_order(monkeypatch) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    def delayed(worker: Path, cache: Path, threshold: float) -> dict[str, float]:
        del worker
        nonlocal active, maximum_active
        index = int(cache.stem)
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep((8 - index) * 0.002)
        with lock:
            active -= 1
        return {metric: index + threshold for metric in parallel.METRICS}

    monkeypatch.setattr(parallel, "score", delayed)
    caches = [Path(f"{index}.npz") for index in range(8)]
    sequential = parallel.score_cache_set(Path("worker.py"), caches, 0.5, 1)
    concurrent = parallel.score_cache_set(Path("worker.py"), caches, 0.5, 8)
    assert concurrent == sequential
    assert list(concurrent) == [path.name for path in caches]
    assert maximum_active > 1


def test_parallel_worker_bounds_fail_closed() -> None:
    for workers in (0, 33):
        try:
            parallel.score_cache_set(Path("worker.py"), [], 0.5, workers)
        except ValueError as error:
            assert "parallel workers" in str(error)
        else:
            raise AssertionError("unfrozen worker count was accepted")
