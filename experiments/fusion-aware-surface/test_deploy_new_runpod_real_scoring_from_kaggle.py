"""Fail-closed checks for the fresh CPU-only verified-parallel deployer."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_new_deployer_is_cpu_only_fresh_and_credential_ephemeral() -> None:
    source = (HERE / "deploy_new_runpod_real_scoring_from_kaggle.py").read_text(
        encoding="utf-8"
    )
    assert "resume_pod" not in source
    assert "create_pod" not in source
    assert "validate_pod(pod)" in source
    assert 'retry.get("parallel_workers") != 32' in source
    assert 'additional_authorized_cap_usd") != 21.0' in source
    assert "kaggle-credential-v1/access_token" in source
    assert "real-scoring-work-v12" in source
    assert "runpod.stop_pod(pod_id)" in source
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y rsync" in source
