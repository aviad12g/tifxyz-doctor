from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_runpod_retry_uses_result_blind_high_cpu_fanout() -> None:
    launcher = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    freezer = (HERE / "freeze_runpod_real_scoring_retry.py").read_text(encoding="utf-8")
    assert "REAL_PARALLEL_WORKERS = 32" in launcher
    assert 'str(REAL_PARALLEL_WORKERS)' in launcher
    assert '"absolute_cap_usd": 3.50' in freezer
    assert '"compute_cutoff_usd": 3.25' in freezer
    assert '"guard_seconds": 300' in freezer
    assert '"failed_or_sealed_results_used_to_design_correction": False' in freezer


def test_runpod_retry_is_operational_in_scientific_projection() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text(encoding="utf-8")
    assert '"result_blind_runpod_real_scoring_retry"' in source
