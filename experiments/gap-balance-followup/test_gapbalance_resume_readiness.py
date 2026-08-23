from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_module("audit_gapbalance_supplemental_controls", "audit_gapbalance_supplemental_controls.py")
READINESS = load_module("verify_gapbalance_resume_readiness", "verify_gapbalance_resume_readiness.py")


def write_controls(path: Path, *, empty: bool = False, one_slab: bool = False) -> list[dict]:
    rows = []
    for index in range(60):
        slab = 0 if one_slab else index % 4
        z = 65 + (11011 - 65) * index / 59
        rows.append(
            {
                "arm": "control",
                "file": f"control/control_{index:03d}.npz",
                "sha256": hashlib.sha256(f"control-{index}".encode()).hexdigest(),
                "ct_empty_frac": 0.2 if empty else 0.05,
                "source_slab": slab,
                "center_z_level1": z,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_supplemental_controls_metadata_audit_passes_without_npz_access(tmp_path):
    manifest = tmp_path / "CONTROLS.jsonl"
    write_controls(manifest)
    requirements = AUDIT.load_requirements(ROOT / "GAPBALANCE_SUPPLEMENTAL_CONTROL_REQUIREMENTS.json")
    result = AUDIT.audit_controls(
        requirements,
        manifest,
        dataset_ref="author/control-supplement",
        dataset_version=1,
        dataset_id=123,
        expected_manifest_sha256=AUDIT.sha256_file(manifest),
        expected_manifest_bytes=manifest.stat().st_size,
    )
    assert result["inventory"]["eligible_controls"] == 60
    assert result["inventory"]["distinct_eligible_source_slabs"] == 4
    assert result["quarantine_superseded"] is False
    assert result["npz_ct_labels_predictions_probabilities_panels_or_endpoints_opened"] is False


def test_supplement_fails_on_empty_controls_or_single_slab(tmp_path):
    requirements = AUDIT.load_requirements(ROOT / "GAPBALANCE_SUPPLEMENTAL_CONTROL_REQUIREMENTS.json")
    for name, kwargs, match in (
        ("empty", {"empty": True}, "too few eligible controls"),
        ("one-slab", {"one_slab": True}, "too few distinct"),
    ):
        manifest = tmp_path / f"{name}.jsonl"
        write_controls(manifest, **kwargs)
        with pytest.raises(AUDIT.AuditError, match=match):
            AUDIT.audit_controls(
                requirements,
                manifest,
                dataset_ref="author/control-supplement",
                dataset_version=1,
                dataset_id=None,
                expected_manifest_sha256=AUDIT.sha256_file(manifest),
                expected_manifest_bytes=manifest.stat().st_size,
            )


def test_supplement_requirements_payload_is_tamper_evident(tmp_path):
    payload = json.loads((ROOT / "GAPBALANCE_SUPPLEMENTAL_CONTROL_REQUIREMENTS.json").read_text())
    payload["control_rules"]["minimum_eligible_controls"] = 1
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AUDIT.AuditError, match="payload SHA-256 mismatch"):
        AUDIT.load_requirements(path)


def test_blocked_resume_package_is_ready_but_non_executable():
    result = READINESS.verify(ROOT)
    assert result["status"] == "BLOCKED_PACKAGE_READY_FOR_LATER_HOLDOUT_REASSESSMENT"
    assert result["completed_cache_jobs"] == 5
    assert result["pending_cache_jobs"] == 8
    assert result["provider_actions_executed"] is False
    assert result["scientific_endpoints_scored"] is False
    assert result["confirmation_outputs_inspected"] is False
    assert result["quarantine_active"] is True


def test_resume_draft_is_tamper_evident(tmp_path):
    for source in ROOT.iterdir():
        if source.is_file():
            (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "GAPBALANCE_KAGGLE_RESUME_DRAFT.json"
    payload = json.loads(path.read_text())
    payload["activation"]["launch_permitted"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(READINESS.ReadinessError, match="embedded payload mismatch"):
        READINESS.verify(tmp_path)
