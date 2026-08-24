from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_projection_allowlist_contains_cpu_records() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text(encoding="utf-8")
    assert '"result_blind_runpod_cpu_real_scoring"' in source
    assert '"result_blind_runpod_cpu_projection_correction"' in source
    assert '"result_blind_runpod_cpu_exact_scorer_correction"' in source
    assert '"runpod_cpu_scoring_asset_stager"' in source


def test_freezer_records_result_blind_failure_and_scientific_gate() -> None:
    source = (HERE / "freeze_runpod_cpu_projection_correction.py").read_text(
        encoding="utf-8"
    )
    assert '"fixed_panel_renderer_started": False' in source
    assert '"official_metric_scorer_started": False' in source
    assert '"cache_npz_opened_or_inspected": False' in source
    assert '"scientific_projection_comparison_retained": True' in source
