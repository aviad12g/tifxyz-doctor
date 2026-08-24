from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_legacy_kaggle_json_retry_is_ephemeral_cpu_only_and_result_blind() -> None:
    source = (HERE / "freeze_new_runpod_legacy_kaggle_credentials_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"remote_filename": "kaggle.json"' in source
    assert '"remote_mode": "0600"' in source
    assert '"kaggle_api_token_environment_absent": True' in source
    assert '"credential_value_or_hash_frozen_logged_or_recorded": False' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"gpu_count": 0' in source
