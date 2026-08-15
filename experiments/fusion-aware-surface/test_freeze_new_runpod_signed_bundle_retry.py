from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_signed_bundle_retry_keeps_credentials_local_and_science_sealed() -> None:
    source = (HERE / "freeze_new_runpod_signed_bundle_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"long_lived_kaggle_credential_leaves_local_machine": False' in source
    assert '"signed_url_value_or_hash_frozen_logged_or_recorded": False' in source
    assert '"remote_preflight_range_bytes": "0-0"' in source
    assert '"signed_url_file_removed_before_bundle_request": True' in source
    assert '"downloaded_files_rehashed_without_npz_parsing": True' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"gpu_count": 0' in source
