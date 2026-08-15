from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_launcher_diagnostic_freezer_is_result_blind_and_budget_bounded() -> None:
    source = (HERE / "freeze_new_runpod_launcher_diagnostic_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 15.0' in source
    assert '"scientific_stdout_projected": False' in source
    assert '"maximum_projected_lines": 3' in source
    assert '"parallel_workers": 32' in source
    assert '"gpu_count": 0' in source
