import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_historical_exact_retry_is_preserved_and_verified_parallel_is_active() -> None:
    launcher = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    freezer = (HERE / "freeze_runpod_real_scoring_retry.py").read_text(encoding="utf-8")
    plan = json.loads((HERE / "heldout_execution_plan.json").read_text())
    assert "REAL_PARALLEL_WORKERS = 32" in launcher
    assert '"--parallel-workers"' in launcher
    assert plan["one_shot_scorers"]["real"]["script"]["sha256"] == (
        "f280c5ed7c74df27d9986757e05109d358caf429e23e4698fb44c91616387f7c"
    )
    assert plan["result_blind_runpod_cpu_exact_scorer_correction"]["decision"][
        "parallel_scorer_used_for_retry"
    ] is False
    assert plan["result_blind_verified_parallel_real_scoring"][
        "exact_predecessor_scorer"
    ]["sha256"] == (
        "3529b8213237a60d392ffec04efca602988b3242f6af8cadc87423e8e224bb79"
    )
    assert plan["result_blind_verified_parallel_real_scoring"][
        "parallel_workers"
    ] == 32
    assert '"absolute_cap_usd": 3.50' in freezer
    assert '"compute_cutoff_usd": 3.25' in freezer
    assert '"guard_seconds": 300' in freezer
    assert '"failed_or_sealed_results_used_to_design_correction": False' in freezer


def test_runpod_retry_is_operational_in_scientific_projection() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text(encoding="utf-8")
    assert '"result_blind_runpod_real_scoring_retry"' in source
