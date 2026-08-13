from __future__ import annotations

import json
import sys
from pathlib import Path

import freeze_heldout_cache_delivery as freezer
import pytest

RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)
PUBLIC_COMMIT = "a" * 40


def _canonical(payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = freezer.sha256_bytes(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    )
    return result


def _jobs(mode: str) -> list[dict]:
    records = []
    for run in RUNS:
        if mode == "real_test_cache":
            records.append(
                {
                    "job_id": f"real-test-{run.replace('_', '-')}",
                    "mode": mode,
                    "run": run,
                    "expected_cache_manifest": f"cache_manifest_{run}_test.json",
                }
            )
        else:
            records.append(
                {
                    "job_id": f"synthetic-{run.replace('_', '-')}-all-shards",
                    "mode": mode,
                    "run": run,
                    "expected_cache_manifests": [
                        f"synthetic_manifest_{run}_shard{shard:02d}.json" for shard in range(10)
                    ],
                }
            )
    return records


def _write_plan(tmp_path: Path) -> tuple[Path, dict]:
    collector = tmp_path.parent / "collect_heldout_delivery_inputs.py"
    if not collector.is_file():
        collector = Path(__file__).resolve().with_name("collect_heldout_delivery_inputs.py")
    threshold_binding = {
        "frozen_thresholds": {"payload_sha256": "f" * 64},
        "kernel_id": "aviadcohen1/vesuvius-fusion-aware-freeze-thresholds",
        "kernel_version": 6,
    }
    plan = _canonical(
        {
            "schema_version": "1.0",
            "status": "held-out execution plan frozen before held-out inference",
            "threshold_binding": threshold_binding,
            "heldout_cache_launcher": {
                "file": "heldout_cache_launcher.py",
                "bytes": 123,
                "sha256": "b" * 64,
            },
            "heldout_delivery_collector": {
                "file": collector.name,
                "bytes": collector.stat().st_size,
                "sha256": freezer.sha256_file(collector),
            },
            "heldout_delivery_freezer": {
                "file": Path(freezer.__file__).resolve().name,
                "bytes": Path(freezer.__file__).resolve().stat().st_size,
                "sha256": freezer.sha256_file(Path(freezer.__file__).resolve()),
            },
            "real_test_jobs": _jobs("real_test_cache"),
            "synthetic_ray_jobs": _jobs("synthetic_ray_cache"),
        }
    )
    path = tmp_path / "heldout_execution_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, plan


def _write_index(root: Path, job: dict, plan_path: Path, plan: dict) -> Path:
    destination = root / job["job_id"]
    destination.mkdir()
    manifest_names = (
        [job["expected_cache_manifest"]]
        if job["mode"] == "real_test_cache"
        else job["expected_cache_manifests"]
    )
    cache_count = 38 if job["mode"] == "real_test_cache" else 100
    index = _canonical(
        {
            "schema_version": "1.0",
            "status": "one publicly planned held-out cache job sealed without scoring",
            "job": job,
            "public_execution_plan": {
                "commit": PUBLIC_COMMIT,
                "file_sha256": freezer.sha256_file(plan_path),
                "payload_sha256": plan["payload_sha256"],
            },
            "threshold_binding": plan["threshold_binding"],
            "threshold_payload_sha256": "f" * 64,
            "launcher": plan["heldout_cache_launcher"],
            "job_config": freezer.expected_job_config_identity(
                job_id=job["job_id"],
                public_plan_commit=PUBLIC_COMMIT,
                public_plan_file_sha256=freezer.sha256_file(plan_path),
            ),
            "cache_manifests": [
                {
                    "file": name,
                    "bytes": 1,
                    "sha256": f"{index + 1:064x}",
                    "payload_sha256": f"{index + 101:064x}",
                }
                for index, name in enumerate(manifest_names)
            ],
            "sealed_cache_files": [
                (
                    {
                        "file": f"cache_{index:03d}.npz",
                        "bytes": index + 1,
                        "sha256": f"{index + 201:064x}",
                        "source_image": f"source_{index:03d}.tif",
                    }
                    if job["mode"] == "real_test_cache"
                    else {
                        "file": f"cache_{index:03d}.npz",
                        "bytes": index + 1,
                        "sha256": f"{index + 201:064x}",
                        "name": f"cell_{index:03d}",
                        "kind": (
                            "primary" if index % 2 == 0 else "single_sheet_control"
                        ),
                        "seed": 300 + index % 5,
                        "pitch_um": 170.0 + 30.0 * (index % 4),
                        "papyrus": 35 + 15 * (index % 4),
                        "kollesis": bool(index % 2),
                    }
                )
                for index in range(cache_count)
            ],
            "scientific_gate": {
                "thresholds_were_publicly_frozen_before_this_job": True,
                "held_out_inference_executed": True,
                "scientific_endpoints_scored": False,
                "scientific_endpoints_printed": False,
                "manual_threshold_override_used": False,
                "cache_payload_opened_or_inspected_by_launcher": False,
            },
        }
    )
    path = destination / "heldout_job_index.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_identity_schema_matches_launcher_records() -> None:
    real = {
        "file": "patch.npz",
        "bytes": 1,
        "sha256": "a" * 64,
        "source_image": "patch.tif",
    }
    synthetic = {
        "file": "cell.npz",
        "bytes": 1,
        "sha256": "b" * 64,
        "name": "cell",
        "kind": "primary",
        "seed": 300,
        "pitch_um": 260.0,
        "papyrus": 65,
        "kollesis": True,
    }
    freezer.validate_identity_record(
        real, expected_count=38, mode="real_test_cache"
    )
    freezer.validate_identity_record(
        synthetic, expected_count=100, mode="synthetic_ray_cache"
    )

    missing_source = dict(real)
    missing_source.pop("source_image")
    with pytest.raises(RuntimeError, match="identity schema mismatch"):
        freezer.validate_identity_record(
            missing_source, expected_count=38, mode="real_test_cache"
        )

    extra_synthetic = dict(synthetic, unexpected=True)
    with pytest.raises(RuntimeError, match="identity schema mismatch"):
        freezer.validate_identity_record(
            extra_synthetic, expected_count=100, mode="synthetic_ray_cache"
        )

    invalid_geometry = dict(synthetic, pitch_um=False)
    with pytest.raises(RuntimeError, match="geometry is invalid"):
        freezer.validate_identity_record(
            invalid_geometry, expected_count=100, mode="synthetic_ray_cache"
        )


def test_delivery_freezer_accepts_only_canonical_collector_record(
    tmp_path: Path, monkeypatch
) -> None:
    plan_path, plan = _write_plan(tmp_path)
    jobs = freezer.expected_jobs(plan)
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    sources = []
    for job in jobs:
        index_path = _write_index(indexes, job, plan_path, plan)
        sources.append(
            {
                "job_id": job["job_id"],
                "kernel_id": freezer.expected_kernel_id(job["job_id"]),
                "kernel_version": 1,
                "job_index": str(index_path),
            }
        )
    records = _canonical(
        {
            "schema_version": "1.0",
            "status": "all held-out delivery indexes and manifests collected without NPZ",
            "public_plan": {
                "commit": PUBLIC_COMMIT,
                "file": plan_path.name,
                "bytes": plan_path.stat().st_size,
                "sha256": freezer.sha256_file(plan_path),
                "payload_sha256": plan["payload_sha256"],
            },
            "package_index": {},
            "queue_receipt": {},
            "jobs": sources,
            "collector": plan["heldout_delivery_collector"],
            "scientific_gate": {
                "all_latest_kernel_versions_match_acceptance_receipt": True,
                "all_kernel_sources_match_accepted_packages": True,
                "npz_probability_caches_downloaded": False,
                "kernel_logs_opened_or_read": False,
                "scientific_endpoints_opened_or_read": False,
            },
        }
    )
    records_path = tmp_path / "delivery_source_records.json"
    records_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "heldout_cache_delivery.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze_heldout_cache_delivery.py",
            "--plan",
            str(plan_path),
            "--public-plan-commit",
            PUBLIC_COMMIT,
            "--records",
            str(records_path),
            "--out",
            str(output),
        ],
    )
    assert freezer.main() == 0
    delivery = freezer.load_json(output)
    freezer.canonical_payload_sha256(delivery)
    assert delivery["counts"] == {
        "jobs": 14,
        "real_jobs": 7,
        "synthetic_jobs": 7,
        "real_probability_caches": 266,
        "synthetic_ray_caches": 700,
    }

    first_index = Path(sources[0]["job_index"])
    tampered = freezer.load_json(first_index)
    tampered["job_config"]["bytes"] += 1
    content = dict(tampered)
    content.pop("payload_sha256")
    tampered["payload_sha256"] = freezer.sha256_bytes(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )
    first_index.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedded job-config identity mismatch"):
        freezer.validate_job_index(
            path=first_index,
            job=jobs[0],
            plan=plan,
            public_plan_commit=PUBLIC_COMMIT,
            public_plan_file_sha256=freezer.sha256_file(plan_path),
        )
