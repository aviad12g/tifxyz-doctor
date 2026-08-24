from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "collect_runpod_primary_outputs", HERE / "collect_runpod_primary_outputs.py"
)
collector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collector)


def complete_status() -> tuple[dict, list[str], str]:
    job_ids = [f"job-{index}" for index in range(7)]
    plan_sha = "a" * 64
    status = {
        "state": "COMPLETE",
        "scientific_outputs_inspected": False,
        "plan_payload_sha256": plan_sha,
        "jobs": {
            job_id: {"gpu_index": index, "returncode": 0, "state": "COMPLETE"}
            for index, job_id in enumerate(job_ids)
        },
    }
    return status, job_ids, plan_sha


def test_complete_status_requires_all_seven_successful_jobs() -> None:
    status, job_ids, plan_sha = complete_status()
    collector.validate_complete_status(status, job_ids, plan_sha)
    status["jobs"][job_ids[3]]["returncode"] = 1
    with pytest.raises(RuntimeError, match="not complete"):
        collector.validate_complete_status(status, job_ids, plan_sha)


def test_complete_status_rejects_running_or_unblinded_delivery() -> None:
    status, job_ids, plan_sha = complete_status()
    status["state"] = "RUNNING"
    with pytest.raises(RuntimeError, match="not COMPLETE"):
        collector.validate_complete_status(status, job_ids, plan_sha)
    status["state"] = "COMPLETE"
    status["scientific_outputs_inspected"] = True
    with pytest.raises(RuntimeError, match="blind gate"):
        collector.validate_complete_status(status, job_ids, plan_sha)


def test_collector_never_imports_npz_readers() -> None:
    source = (HERE / "collect_runpod_primary_outputs.py").read_text(encoding="utf-8")
    assert "numpy" not in source
    assert "np.load" not in source
    assert "zipfile" not in source
