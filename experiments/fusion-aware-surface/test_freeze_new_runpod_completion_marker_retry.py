"""Structural checks for the KaggleHub completion-marker retry."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_completion_marker_retry_is_result_blind_and_budgeted() -> None:
    source = (HERE / "freeze_new_runpod_completion_marker_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 19.0' in source
    assert '"reserve_usd": 1.0' in source
    assert "bundle.complete" in source
    assert '"expected_marker_bytes": 0' in source
    assert '"reject_any_other_completion_entry": True' in source
    assert '"real_scorer_started": False' in source
    assert '"npz_payloads_opened_or_parsed": False' in source
    assert '"gpu_count": 0' in source
