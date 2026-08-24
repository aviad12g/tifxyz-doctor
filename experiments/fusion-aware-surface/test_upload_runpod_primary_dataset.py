from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "upload_runpod_primary_dataset", HERE / "upload_runpod_primary_dataset.py"
)
uploader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(uploader)

STAGER_SPEC = importlib.util.spec_from_file_location(
    "stage_runpod_primary_dataset", HERE / "stage_runpod_primary_dataset.py"
)
stager = importlib.util.module_from_spec(STAGER_SPEC)
assert STAGER_SPEC.loader is not None
STAGER_SPEC.loader.exec_module(stager)


def write_hashed(path: Path, payload: dict) -> dict:
    value = dict(payload)
    value["payload_sha256"] = stager.canonical_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def test_parse_ledger_accepts_canonical_nested_records(tmp_path: Path) -> None:
    ledger = tmp_path / "DATASET_SHA256SUMS"
    ledger.write_text(
        f"{'a' * 64}  primary/job/heldout_job_index.json\n"
        f"{'b' * 64}  primary/job/run/cache.npz\n",
        encoding="utf-8",
    )
    assert uploader.parse_ledger(ledger) == {
        "primary/job/heldout_job_index.json": "a" * 64,
        "primary/job/run/cache.npz": "b" * 64,
    }


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../b", "./file"])
def test_safe_relative_rejects_escape_paths(relative: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        uploader.safe_relative(relative)


def test_uploader_is_result_blind_and_uses_nested_transport() -> None:
    source = (HERE / "upload_runpod_primary_dataset.py").read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "numpy" not in source
    assert "zipfile" not in source
    assert "MAX_FILES_TO_UPLOAD = 1000" in source
    assert "dataset_upload(" in source


def test_full_seven_job_staging_and_upload_validation_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = []
    for job_number in range(7):
        job_id = f"synthetic-fixture-{job_number}"
        root = tmp_path / "source" / job_id
        run_name = f"run_{job_number}"
        run_root = root / run_name
        run_root.mkdir(parents=True)
        manifests = []
        for manifest_number in range(10):
            path = root / f"cache_manifest_{manifest_number:02d}.json"
            payload = write_hashed(
                path,
                {"job_id": job_id, "manifest_number": manifest_number},
            )
            manifests.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": stager.sha256_file(path),
                    "payload_sha256": payload["payload_sha256"],
                }
            )
        caches = []
        for cache_number in range(100):
            path = run_root / f"cache_{cache_number:03d}.npz"
            path.write_bytes(f"opaque-{job_number}-{cache_number}".encode())
            caches.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": stager.sha256_file(path),
                }
            )
        index_path = root / "heldout_job_index.json"
        index = write_hashed(
            index_path,
            {
                "job": {"run": run_name},
                "cache_manifests": manifests,
                "sealed_cache_files": caches,
            },
        )
        jobs.append(
            {
                "job_id": job_id,
                "job_index": str(index_path),
                "job_index_identity": {
                    "bytes": index_path.stat().st_size,
                    "sha256": stager.sha256_file(index_path),
                    "payload_sha256": index["payload_sha256"],
                },
            }
        )
    records_path = tmp_path / "runpod_records.json"
    write_hashed(
        records_path,
        {
            "status": "seven authoritative RunPod primary deliveries verified without opening NPZ",
            "counts": {
                "jobs": 7,
                "cache_manifests": 70,
                "sealed_cache_files": 700,
            },
            "scientific_gate": {
                "all_primary_jobs_complete": True,
                "all_copied_cache_hashes_verified": True,
                "npz_cache_payloads_opened_or_inspected": False,
                "scientific_endpoints_opened_or_read": False,
                "runpod_is_authoritative_primary": True,
            },
            "jobs": jobs,
        },
    )
    dataset_root = tmp_path / "dataset"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_runpod_primary_dataset.py",
            "--records",
            str(records_path),
            "--out",
            str(dataset_root),
        ],
    )
    assert stager.main() == 0
    manifest, ledger = uploader.validate_staging(dataset_root)
    assert manifest["counts"] == {
        "jobs": 7,
        "cache_manifests": 70,
        "sealed_cache_files": 700,
        "staged_scientific_files": 777,
    }
    assert len(ledger) == 777
    assert len([path for path in dataset_root.rglob("*") if path.is_file()]) == 780
