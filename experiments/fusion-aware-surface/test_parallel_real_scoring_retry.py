#!/usr/bin/env python3
"""Result-blind tests for the now-proven concurrent real scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXACT_SCORER_SHA256 = "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79"


def test_exact_predecessor_is_preserved_and_verified_parallel_is_active() -> None:
    scorer_path = HERE / "score_real_test.py"
    source = scorer_path.read_text(encoding="utf-8")
    assert hashlib.sha256(scorer_path.read_bytes()).hexdigest() == EXACT_SCORER_SHA256
    assert "score_cache_set" not in source
    assert "parallel-workers" not in source

    plan = json.loads((HERE / "heldout_execution_plan.json").read_text())
    assert plan["one_shot_scorers"]["real"]["script"] == {
        "file": "score_real_test_parallel.py",
        "sha256": "f280c5ed7c74df27d9986757e05109d358caf429e23e4698fb44c91616387f7c",
    }
    decision = plan["result_blind_runpod_cpu_exact_scorer_correction"]["decision"]
    assert decision["parallel_scorer_used_for_retry"] is False
    assert decision["exact_original_real_scorer_restored"]["sha256"] == (
        EXACT_SCORER_SHA256
    )
    verified = plan["result_blind_verified_parallel_real_scoring"]
    assert verified["exact_predecessor_scorer"] == {
        "file": "score_real_test.py",
        "sha256": EXACT_SCORER_SHA256,
    }
    assert verified["parallel_workers"] == 32
    assert verified["scientific_contract"][
        "only_independent_subprocess_scheduling_changed"
    ] is True


def test_launcher_runs_panel_fail_fast_before_long_real_scorer() -> None:
    source = Path(__file__).with_name("one_shot_scoring_launcher.py").read_text(
        encoding="utf-8"
    )
    main = source[source.index("def main() -> int:") :]
    panel = main.index("panel_invocation = execute_real_panels(")
    scoring = main.index("result_path, invocation = execute_scorer(")
    assert panel < scoring
    assert "REAL_PARALLEL_WORKERS = 32" in source
    assert 'command.extend(["--parallel-workers", str(REAL_PARALLEL_WORKERS)])' in source
    assert "sys.stderr.buffer.write(completed.stderr)" in source
