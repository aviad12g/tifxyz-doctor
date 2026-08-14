from __future__ import annotations

import json
from pathlib import Path

import freeze_runpod_real_panel_provenance_retry as freeze


HERE = Path(__file__).resolve().parent


def test_panel_provenance_retry_preserves_scientific_contract() -> None:
    predecessor = json.loads(
        (HERE / "runpod_real_scoring_metric_verifier_plan.json").read_text()
    )
    corrected = freeze.load_hashed(
        HERE / "runpod_real_scoring_panel_provenance_plan.json"
    )
    ignored = {
        "payload_sha256",
        "public_execution_plan",
        "public_cache_delivery",
        "embedded_real_launcher",
        "result_blind_real_panel_index_provenance_retry",
    }
    assert {key: value for key, value in corrected.items() if key not in ignored} == {
        key: value for key, value in predecessor.items() if key not in ignored
    }
    assert corrected["scientific_gate"] == predecessor["scientific_gate"]
    assert corrected["public_execution_plan"] == freeze.PUBLIC_PLAN
    assert corrected["public_cache_delivery"] == freeze.PUBLIC_DELIVERY
    assert corrected["embedded_real_launcher"] == freeze.EMBEDDED_LAUNCHER
    record = corrected["result_blind_real_panel_index_provenance_retry"]
    assert record == freeze.correction_record()
    assert record["scientific_gate"] == {
        "cache_model_metric_threshold_seed_panel_selection_endpoint_gate_aggregation_or_claim_changed": False,
        "test_time_tuning_permitted": False,
        "result_probability_endpoint_panel_or_npz_opened": False,
    }
