"""Structural checks for the manifest file-identity retry."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_manifest_identity_retry_is_result_blind_and_budgeted() -> None:
    source = (HERE / "freeze_new_runpod_manifest_identity_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 18.0' in source
    assert '"reserve_usd": 1.0' in source
    assert '"file_identity_fields": ["file", "bytes", "sha256"]' in source
    assert '"payload_sha256_validated_separately": True' in source
    assert '"manifest_bytes_changed": False' in source
    assert '"npz_downloaded_opened_or_parsed": False' in source
    assert '"real_scorer_started": False' in source
    assert '"gpu_count": 0' in source
