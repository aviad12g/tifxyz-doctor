from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_scorer_exception_retry_freezer_preserves_science_and_budget() -> None:
    source = (HERE / "freeze_new_runpod_scorer_exception_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 12.0' in source
    assert '"only_bounded_stderr_exception_projection_changed": True' in source
    assert '"scientific_stdout_projected": False' in source
    assert '"parallel_workers": 32' in source
    assert '"gpu_count": 0' in source
