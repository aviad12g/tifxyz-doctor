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
