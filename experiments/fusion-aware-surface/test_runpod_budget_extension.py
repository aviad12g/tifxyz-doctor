from __future__ import annotations

import json
from pathlib import Path

import pytest

import apply_runpod_budget_extension as extension_tool


def canonical(payload: dict) -> dict:
    result = dict(payload)
    result["payload_sha256"] = extension_tool.canonical_sha256(result)
    return result


def fixture_payloads(tmp_path: Path) -> tuple[Path, Path]:
    extension = canonical(
        {
            "authorization": {
                "additional_authorized_usd": 15.0,
                "authorized_at_utc": "2026-08-12T18:51:58Z",
                "new_authorized_total_usd": 37.0,
                "user_reaffirmed_prior_total_usd": 22.0,
            },
            "budget": {
                "maximum_accepted_gpu_hourly_price_usd": 0.4,
                "new_billing_cutoff_usd": 35.5,
                "new_hard_total_cap_usd": 37.0,
                "non_compute_reserve_usd": 1.5,
                "on_cutoff": "stop and preserve",
                "prior_public_receipt_billing_cutoff_usd": 18.5,
                "prior_public_receipt_hard_total_cap_usd": 20.0,
            },
            "frozen_primary": {
                "active_aggregate_hourly_rate_usd": 2.38,
                "job_count": 7,
                "original_plan_payload_sha256": "a" * 64,
                "original_public_replacement_commit": "b" * 40,
                "pod_id": "pod",
            },
            "scientific_protocol": {
                "cache_or_endpoint_values_opened_before_extension": False,
                "claim_or_selection_changed": False,
                "datasets_changed": False,
                "gates_changed": False,
                "hardware_or_worker_layout_changed": False,
                "models_or_checkpoints_changed": False,
                "panels_changed": False,
                "seeds_changed": False,
                "thresholds_changed": False,
            },
            "status": "result-blind operational budget extension authorized before primary completion",
        }
    )
    receipt = {
        "billing_cutoff_usd": 18.5,
        "hard_total_cap_usd": 20.0,
        "plan_payload_sha256": "a" * 64,
        "pods": [{"id": "pod", "jobs": [str(index) for index in range(7)]}],
        "public_replacement_commit": "b" * 40,
        "scientific_outputs_inspected": False,
        "status": "PRIMARY_EXECUTOR_RUNNING_SEALED",
        "total_hourly_rate_usd": 2.38,
    }
    extension_path = tmp_path / "extension.json"
    receipt_path = tmp_path / "receipt.json"
    extension_path.write_text(json.dumps(extension), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return extension_path, receipt_path


def test_applies_only_budget_fields_and_records_public_commit(tmp_path: Path) -> None:
    extension_path, receipt_path = fixture_payloads(tmp_path)
    before = json.loads(receipt_path.read_text(encoding="utf-8"))
    after = extension_tool.apply(extension_path, receipt_path, "c" * 40)
    assert after["billing_cutoff_usd"] == 35.5
    assert after["hard_total_cap_usd"] == 37.0
    assert after["plan_payload_sha256"] == before["plan_payload_sha256"]
    assert after["pods"] == before["pods"]
    assert after["scientific_outputs_inspected"] is False
    assert after["budget_extensions"][0]["public_extension_commit"] == "c" * 40


def test_is_idempotent_for_same_extension(tmp_path: Path) -> None:
    extension_path, receipt_path = fixture_payloads(tmp_path)
    first = extension_tool.apply(extension_path, receipt_path, "c" * 40)
    second = extension_tool.apply(extension_path, receipt_path, "c" * 40)
    assert second == first


def test_rejects_after_scientific_output_access(tmp_path: Path) -> None:
    extension_path, receipt_path = fixture_payloads(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["scientific_outputs_inspected"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="before scientific output access"):
        extension_tool.apply(extension_path, receipt_path, "c" * 40)


def test_rejects_changed_scientific_protocol(tmp_path: Path) -> None:
    extension_path, receipt_path = fixture_payloads(tmp_path)
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    extension["scientific_protocol"]["thresholds_changed"] = True
    extension["payload_sha256"] = extension_tool.canonical_sha256(extension)
    extension_path.write_text(json.dumps(extension), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changes or opens scientific state"):
        extension_tool.apply(extension_path, receipt_path, "c" * 40)
