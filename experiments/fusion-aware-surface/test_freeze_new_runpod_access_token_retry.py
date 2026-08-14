"""Structural checks for the current Kaggle access-token retry."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_access_token_retry_is_ephemeral_result_blind_and_budgeted() -> None:
    source = (HERE / "freeze_new_runpod_access_token_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 18.5' in source
    assert '"reserve_usd": 1.0' in source
    assert '"credential_kind": "Kaggle access token"' in source
    assert '"credential_value_or_hash_frozen_logged_or_recorded": False' in source
    assert '"remove_immediately_after_download_attempt": True' in source
    assert '"failure_status_code": 403' in source
    assert '"real_scorer_started": False' in source
    assert '"gpu_count": 0' in source
