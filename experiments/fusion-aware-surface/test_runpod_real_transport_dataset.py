from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stager = load("stage_runpod_real_transport_dataset")
uploader = load("upload_runpod_real_transport_dataset")
puller = load("pull_runpod_real_transport_dataset")
wrapper = load("runpod_prepare_and_execute_from_kaggle")
deployer = load("deploy_runpod_real_scoring_from_kaggle")
freezer = load("freeze_runpod_real_kaggle_transport_retry")


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/../b", "./file", ""])
def test_safe_relative_rejects_unsafe_paths(value: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        stager.safe_relative(value)
    with pytest.raises(RuntimeError, match="unsafe"):
        uploader.safe_relative(value)
    with pytest.raises(RuntimeError, match="unsafe"):
        puller.safe_relative(value)


def test_transport_is_private_nested_and_result_blind() -> None:
    stage_source = (HERE / "stage_runpod_real_transport_dataset.py").read_text(encoding="utf-8")
    upload_source = (HERE / "upload_runpod_real_transport_dataset.py").read_text(encoding="utf-8")
    pull_source = (HERE / "pull_runpod_real_transport_dataset.py").read_text(encoding="utf-8")
    assert "np.load" not in stage_source + upload_source + pull_source
    assert "zipfile" not in stage_source + upload_source
    assert '"isPrivate": True' in stage_source
    assert "MAX_FILES_TO_UPLOAD = 500" in upload_source
    assert "dataset_upload(" in upload_source
    assert "versions/{DATASET_VERSION}" in pull_source
    assert "dataset_download(" in pull_source
    assert "remove_credentials(args.credentials)" in pull_source


def test_exact_frozen_counts() -> None:
    assert stager.EXPECTED_COUNTS == {"jobs": 7, "manifests": 7, "sealed_npz_caches": 266}
    assert stager.EXPECTED_SOURCE_FILES == uploader.EXPECTED_SOURCE_FILES == 281
    assert stager.EXPECTED_TOTAL_FILES == uploader.EXPECTED_TOTAL_FILES == 284
    assert stager.DATASET_ID == uploader.DATASET_ID
    assert puller.EXPECTED_SOURCE_FILES == 281
    assert puller.EXPECTED_TOTAL_FILES == 284
    assert puller.EXPECTED_TOTAL_BYTES == 5_791_288_517
    assert puller.DATASET_ID == uploader.DATASET_ID
    assert puller.DATASET_VERSION == 1


def test_cpu_retry_transport_is_result_blind_and_zero_gpu() -> None:
    sources = "\n".join(
        (HERE / name).read_text(encoding="utf-8")
        for name in (
            "pull_runpod_real_transport_dataset.py",
            "runpod_prepare_and_execute_from_kaggle.py",
            "deploy_runpod_real_scoring_from_kaggle.py",
            "freeze_runpod_real_kaggle_transport_retry.py",
        )
    )
    assert "np.load" not in sources
    assert 'runpod.resume_pod(POD_ID, 0)' in sources
    assert '"gpu_count": 0' in sources
    assert '"absolute_cap_usd": 13.5' in sources
    assert "dataset_upload(" not in sources
    assert "submit" not in sources.lower()


def test_ephemeral_credential_is_never_recorded_or_logged() -> None:
    pull_source = (HERE / "pull_runpod_real_transport_dataset.py").read_text(encoding="utf-8")
    wrapper_source = (HERE / "runpod_prepare_and_execute_from_kaggle.py").read_text(encoding="utf-8")
    deploy_source = (HERE / "deploy_runpod_real_scoring_from_kaggle.py").read_text(encoding="utf-8")
    assert "read_text" not in "\n".join(
        line for line in deploy_source.splitlines() if "credentials" in line
    )
    assert "remove_credentials(args.credentials)" in pull_source
    assert "args.credentials.unlink()" in wrapper_source
    assert "str(args.credentials)" not in deploy_source.split("receipt = {", 1)[1].split("}", 1)[0]
