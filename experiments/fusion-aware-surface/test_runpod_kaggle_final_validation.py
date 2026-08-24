from __future__ import annotations

import json
from pathlib import Path

import pytest

import freeze_runpod_kaggle_final_validation as freezer
import validate_runpod_kaggle_final_results as validator


def write_hashed(path: Path, payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = freezer.canonical_sha256(result)
    path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    return result


def test_artifact_record_is_hash_and_path_bound(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    path.write_bytes(b"sealed")
    record = {
        "file": "downloads/real/sealed.json",
        "bytes": path.stat().st_size,
        "sha256": freezer.sha256_file(path),
    }
    validator.validate_artifact_record(record, path, "downloads/real/sealed.json")
    record["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="artifact identity mismatch"):
        validator.validate_artifact_record(record, path, "downloads/real/sealed.json")


def test_bound_source_requires_exact_payload_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source = write_hashed(source_path, {"schema_version": "1.0"})
    plan = {"sources": {"source": freezer.identity(source_path, source)}}
    validator.validate_bound_source(plan, "source", source_path, source)
    tampered = dict(source)
    tampered["payload_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="another source"):
        validator.validate_bound_source(plan, "source", source_path, tampered)


def test_collection_gate_is_complete_and_result_blind() -> None:
    assert validator.COLLECTION_GATE["independent_validation_required_next"] is True
    assert validator.COLLECTION_GATE["results_or_manifests_opened_or_parsed_by_collector"] is False
    assert validator.COLLECTION_GATE["panel_images_opened_or_read_by_collector"] is False


def test_scoring_manifest_plan_identity_omits_transport_fields() -> None:
    public_record = {
        "commit": "a" * 40,
        "file": "heldout_execution_plan.json",
        "bytes": 123,
        "sha256": "b" * 64,
        "payload_sha256": "c" * 64,
    }
    expected = {
        "commit": public_record["commit"],
        "file_sha256": public_record["sha256"],
        "payload_sha256": public_record["payload_sha256"],
    }
    assert set(expected) == {"commit", "file_sha256", "payload_sha256"}
