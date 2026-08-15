from __future__ import annotations

import importlib.util
import json
import sys
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
    assert 'os.environ["KAGGLE_CONFIG_DIR"] = str(args.credentials.parent)' in pull_source
    assert 'os.environ.pop("KAGGLE_API_TOKEN", None)' in pull_source


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


def test_staging_manifest_file_identity_excludes_separately_checked_payload() -> None:
    record = {
        "file": "real_transport_dataset_manifest.json",
        "bytes": 1078,
        "sha256": "a" * 64,
        "payload_sha256": "b" * 64,
    }
    assert puller.file_identity(record) == {
        "file": record["file"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
    }


def test_exact_kagglehub_completion_marker_is_removed(tmp_path: Path) -> None:
    marker = tmp_path / puller.KAGGLEHUB_COMPLETION_MARKER
    marker.parent.mkdir(parents=True)
    marker.touch()
    puller.remove_kagglehub_completion_marker(tmp_path)
    assert not (tmp_path / ".complete").exists()


def test_unexpected_kagglehub_completion_marker_is_rejected(tmp_path: Path) -> None:
    marker = tmp_path / puller.KAGGLEHUB_COMPLETION_MARKER
    marker.parent.mkdir(parents=True)
    marker.touch()
    (marker.parent / "unexpected").touch()
    with pytest.raises(RuntimeError, match="unexpected KaggleHub"):
        puller.remove_kagglehub_completion_marker(tmp_path)


def test_wrapper_propagates_result_blind_child_error(tmp_path: Path) -> None:
    child = tmp_path / "transport-status.json"
    child.write_text(
        json.dumps(
            {
                "state": "ERROR",
                "scientific_outputs_inspected": False,
                "error_type": "RuntimeError",
                "error_message": "exact transport mismatch",
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "child.log").open("wb") as log:
        with pytest.raises(RuntimeError, match="exact transport mismatch"):
            wrapper.run_logged(
                [sys.executable, "-c", "raise SystemExit(1)"],
                log,
                operational_status=child,
            )


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


def test_frozen_private_kaggle_retry_plan() -> None:
    plan_path = HERE / "runpod_real_scoring_private_kaggle_transport_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["payload_sha256"] == puller.canonical_sha256(plan)
    assert plan["public_runpod_commit"] == "84488422d0ee075bae9d429a88f13ebfe2f1bfee"
    assert plan["budget"] == {
        "absolute_cap_usd": 13.5,
        "compute_cutoff_usd": 1.8450102378888895,
        "guard_seconds": 300,
        "maximum_provider_wall_seconds_at_price_ceiling": 5189,
        "maximum_runtime_seconds_before_guard_at_price_ceiling": 4889,
        "on_cutoff": "stop the CPU Pod, preserve sealed volume artifacts, and never score a partial result",
        "prior_guarded_spend_upper_bound_usd": 11.65498976211111,
        "reserve_usd": 0.0,
    }
    transport = plan["private_kaggle_transport"]
    assert transport["dataset_handle"] == (
        "aviadcohen1/vesuvius-fusion-real-heldout-transport-v1/versions/1"
    )
    assert transport["expected_total_files"] == 284
    assert transport["expected_total_bytes"] == 5_791_288_517
    assert transport["kagglehub_wheelhouse_tree"] == {
        "files": 11,
        "bytes": 2_332_451,
        "ledger_sha256": "8ce5a93740006753f13fce6254b1c5487fa70679d9e5f6b5c39a42a9f36ece9e",
    }
    assert plan["private_kaggle_transport_retry"]["scientific_gate"] == {
        "cache_model_metric_threshold_seed_panel_endpoint_gate_aggregation_order_or_claim_changed": False,
        "result_probability_endpoint_panel_or_npz_opened": False,
        "test_time_tuning_permitted": False,
    }
