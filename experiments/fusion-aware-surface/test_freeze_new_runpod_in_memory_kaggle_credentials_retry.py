from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_in_memory_kaggle_retry_is_ephemeral_cpu_only_and_result_blind() -> None:
    source = (HERE / "freeze_new_runpod_in_memory_kaggle_credentials_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"authentication_method": "KaggleHub 1.0.2 in-memory set_kaggle_credentials"' in source
    assert '"credential_file_removed_before_dataset_request": True' in source
    assert '"credential_value_or_hash_frozen_logged_or_recorded": False' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"gpu_count": 0' in source
