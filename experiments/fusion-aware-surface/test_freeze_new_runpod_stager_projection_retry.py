from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_stager_projection_retry_freezer_preserves_science_and_budget() -> None:
    source = (HERE / "freeze_new_runpod_stager_projection_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 13.0' in source
    assert '"only_verified_parallel_scorer_identity_projection_changed": True' in source
    assert '"npz_panel_probability_endpoint_or_result_opened": False' in source
    assert '"parallel_workers": 32' in source
    assert '"gpu_count": 0' in source
