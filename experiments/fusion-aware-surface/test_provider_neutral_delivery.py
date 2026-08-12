from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "freeze_provider_neutral_heldout_delivery",
    HERE / "freeze_provider_neutral_heldout_delivery.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def write_hashed(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["payload_sha256"] = module.canonical_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def test_provider_neutral_freeze_preserves_runpod_authority_and_old_consumer_status(
    tmp_path: Path, monkeypatch
) -> None:
    commit = "a" * 40
    real_jobs = [
        {"job_id": f"real-{index}", "mode": "real_test_cache", "run": f"r{index}"}
        for index in range(7)
    ]
    synthetic_jobs = [
        {"job_id": f"synth-{index}", "mode": "synthetic_ray_cache", "run": f"s{index}"}
        for index in range(7)
    ]
    plan_path = tmp_path / "heldout_execution_plan.json"
    plan = write_hashed(
        plan_path,
        {
            "status": "held-out execution plan frozen before held-out inference",
            "real_test_jobs": real_jobs,
            "synthetic_ray_jobs": synthetic_jobs,
            "threshold_binding": {"frozen": True},
        },
    )
    public = {
        "commit": commit,
        "file": plan_path.name,
        "bytes": plan_path.stat().st_size,
        "sha256": module.sha256_file(plan_path),
        "payload_sha256": plan["payload_sha256"],
    }
    real_path = tmp_path / "real.json"
    real = write_hashed(
        real_path,
        {
            "public_plan": public,
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "execution_origin": {
                        "provider": "Kaggle",
                        "kernel_id": f"aviadcohen1/{job['job_id']}",
                        "kernel_version": 1,
                    },
                    "job_index": str(tmp_path / job["job_id"] / "heldout_job_index.json"),
                }
                for job in real_jobs
            ],
            "scientific_gate": {
                "all_real_kernel_versions_match_acceptance_receipt": True,
                "all_real_kernel_sources_match_accepted_packages": True,
                "npz_probability_caches_downloaded": False,
                "kernel_logs_opened_or_read": False,
                "scientific_endpoints_opened_or_read": False,
            },
        },
    )
    synthetic_path = tmp_path / "synthetic.json"
    synthetic = write_hashed(
        synthetic_path,
        {
            "public_execution_plan": public,
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "execution_origin": {
                        "provider": "RunPod",
                        "pod_id": "pod-1",
                        "public_replacement_commit": "b" * 40,
                        "replacement_plan_payload_sha256": "c" * 64,
                    },
                    "job_index": str(tmp_path / job["job_id"] / "heldout_job_index.json"),
                    "job_index_identity": {"identity": job["job_id"]},
                }
                for job in synthetic_jobs
            ],
            "scientific_gate": {
                "all_primary_jobs_complete": True,
                "all_copied_cache_hashes_verified": True,
                "npz_cache_payloads_opened_or_inspected": False,
                "scientific_endpoints_opened_or_read": False,
                "runpod_is_authoritative_primary": True,
            },
        },
    )
    identities = {
        **{job["job_id"]: {"identity": job["job_id"]} for job in real_jobs},
        **{job["job_id"]: {"identity": job["job_id"]} for job in synthetic_jobs},
    }
    monkeypatch.setattr(
        module.freezer,
        "validate_job_index",
        lambda **kwargs: identities[kwargs["job"]["job_id"]],
    )
    out = tmp_path / "delivery.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_provider_neutral_heldout_delivery.py",
            "--plan",
            str(plan_path),
            "--public-plan-commit",
            commit,
            "--real-records",
            str(real_path),
            "--synthetic-records",
            str(synthetic_path),
            "--out",
            str(out),
        ],
    )
    assert module.main() == 0
    delivered = module.load_hashed(out)
    assert delivered["status"] == (
        "all 14 publicly planned held-out caches sealed before one-shot scoring"
    )
    assert delivered["authority"] == {
        "real_provider": "Kaggle",
        "synthetic_primary_provider": "RunPod",
        "kaggle_synthetic_role": "sealed secondary cross-platform replication",
        "selection_between_synthetic_platforms_permitted": False,
    }
    assert [record["execution_origin"]["provider"] for record in delivered["jobs"]] == (
        ["Kaggle"] * 7 + ["RunPod"] * 7
    )
    assert real["payload_sha256"] != synthetic["payload_sha256"]


def test_provider_neutral_freezer_never_opens_npz_payloads() -> None:
    source = (HERE / "freeze_provider_neutral_heldout_delivery.py").read_text()
    assert "np.load" not in source
    assert "numpy" not in source
    assert "zipfile" not in source
