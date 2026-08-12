from __future__ import annotations

import json
from pathlib import Path

import pytest
import render_real_panels as renderer

PLAN_COMMIT = "a" * 40
DELIVERY_COMMIT = "b" * 40
RUNS = (
    "baseline",
    "control_seed11",
    "control_seed23",
    "control_seed47",
    "gap8_seed11",
    "gap8_seed23",
    "gap8_seed47",
)


def _write_hashed(path: Path, payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = renderer.canonical_sha256(result)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _identity(path: Path, payload: dict | None = None) -> dict:
    record = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": renderer.sha256_file(path),
    }
    if payload is not None:
        record["payload_sha256"] = payload["payload_sha256"]
    return record


def test_panel_context_is_bound_to_public_delivery_and_staging(tmp_path: Path) -> None:
    threshold_path = tmp_path / "frozen_thresholds.json"
    thresholds = _write_hashed(threshold_path, {"schema_version": "1.0"})
    panel_path = tmp_path / "real_panel_manifest.json"
    panels = _write_hashed(
        panel_path,
        {
            "schema_version": "1.0",
            "status": "model-blind; selected from held-out labels before prediction",
            "selected": [{"image": f"image_{index}.tif"} for index in range(4)],
        },
    )
    real_jobs = [
        {
            "job_id": f"real-test-{run.replace('_', '-')}",
            "mode": "real_test_cache",
            "run": run,
        }
        for run in RUNS
    ]
    stager = {"file": "stage_heldout_for_scoring.py", "bytes": 123, "sha256": "c" * 64}
    plan = _write_hashed(
        tmp_path / "heldout_execution_plan.json",
        {
            "schema_version": "1.0",
            "status": "held-out execution plan frozen before held-out inference",
            "real_panel_renderer": _identity(Path(renderer.__file__).resolve()),
            "real_panel_manifest": _identity(panel_path, panels),
            "threshold_binding": {"frozen_thresholds": _identity(threshold_path, thresholds)},
            "one_shot_scoring_stager": stager,
            "real_test_jobs": real_jobs,
        },
    )
    plan_path = tmp_path / "heldout_execution_plan.json"
    delivered_jobs = [
        {
            "job_id": job["job_id"],
            "mode": job["mode"],
            "run": job["run"],
            "kernel_id": f"aviadcohen1/vesuvius-fusion-{job['job_id']}",
            "kernel_version": 1,
            "job_index": {"file": "heldout_job_index.json", "sha256": "d" * 64},
        }
        for job in real_jobs
    ]
    delivery_path = tmp_path / "heldout_cache_delivery_manifest.json"
    delivery = _write_hashed(
        delivery_path,
        {
            "schema_version": "1.0",
            "status": "all 14 publicly planned held-out caches sealed before one-shot scoring",
            "public_execution_plan": {
                "commit": PLAN_COMMIT,
                **_identity(plan_path, plan),
            },
            "threshold_binding": plan["threshold_binding"],
            "jobs": delivered_jobs,
        },
    )
    staged_root = tmp_path / "staged-real"
    staged_root.mkdir()
    staged_jobs = [
        {
            "job_id": job["job_id"],
            "run": job["run"],
            "kernel_id": delivered["kernel_id"],
            "kernel_version": delivered["kernel_version"],
            "job_index": delivered["job_index"],
            "cache_manifest_count": 1,
            "sealed_cache_file_count": 38,
        }
        for job, delivered in zip(real_jobs, delivered_jobs, strict=True)
    ]
    score_index_path = staged_root / "score_input_index.json"
    score_index = _write_hashed(
        score_index_path,
        {
            "schema_version": "1.0",
            "status": "real held-out caches staged from public delivery before one-shot scoring",
            "mode": "real",
            "public_execution_plan": {
                "commit": PLAN_COMMIT,
                **_identity(plan_path, plan),
            },
            "public_cache_delivery": {
                "commit": DELIVERY_COMMIT,
                "payload_sha256": delivery["payload_sha256"],
            },
            "threshold_binding": plan["threshold_binding"],
            "job_order": [job["job_id"] for job in real_jobs],
            "jobs": staged_jobs,
            "counts": {"jobs": 7, "cache_manifests": 7, "sealed_cache_files": 266},
            "scientific_gate": {
                "all_required_cache_jobs_verified": True,
                "cache_delivery_publicly_frozen_before_staging": True,
                "scientific_endpoints_scored": False,
                "scientific_endpoints_printed": False,
                "cache_npz_payloads_opened_or_inspected_by_stager": False,
                "one_shot_scoring_permitted": True,
            },
            "stager": stager,
        },
    )
    _, _, observed_thresholds, observed_panels = renderer.validate_public_context(
        plan_path=plan_path,
        delivery_path=delivery_path,
        score_input_index_path=score_index_path,
        thresholds_path=threshold_path,
        panel_manifest_path=panel_path,
        cache_root=staged_root,
        public_plan_commit=PLAN_COMMIT,
        public_delivery_commit=DELIVERY_COMMIT,
    )
    assert observed_thresholds == thresholds
    assert observed_panels == panels

    tampered = dict(score_index)
    tampered["stager"] = dict(stager) | {"sha256": "0" * 64}
    tampered.pop("payload_sha256")
    tampered_path = staged_root / "tampered_score_input_index.json"
    _write_hashed(tampered_path, tampered)
    with pytest.raises(RuntimeError, match="score-input index provenance mismatch"):
        renderer.validate_public_context(
            plan_path=plan_path,
            delivery_path=delivery_path,
            score_input_index_path=tampered_path,
            thresholds_path=threshold_path,
            panel_manifest_path=panel_path,
            cache_root=staged_root,
            public_plan_commit=PLAN_COMMIT,
            public_delivery_commit=DELIVERY_COMMIT,
        )
