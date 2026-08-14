from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import freeze_real_panel_index_provenance_correction as freeze
import pytest
import stage_heldout_for_scoring as stager


HERE = Path(__file__).resolve().parent
TOOLING_COMMIT = "a" * 40
PLAN_COMMIT = "b" * 40


def test_plan_and_delivery_freeze_exact_operational_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "heldout_execution_plan.json"
    delivery_path = tmp_path / "heldout_cache_delivery_manifest.json"
    alias_path = tmp_path / "heldout_cache_delivery.json"
    current_plan = freeze.load_hashed(HERE / "heldout_execution_plan.json")
    current_plan.pop("result_blind_real_panel_index_provenance_correction", None)
    current_plan["real_panel_renderer"] = freeze.PREDECESSOR_RENDERER
    current_plan["one_shot_scoring_stager"] = freeze.PREDECESSOR_STAGER
    freeze.write_hashed(plan_path, current_plan)
    predecessor_plan = freeze.identity(plan_path) | {
        "commit": freeze.PREDECESSOR_PLAN["commit"],
        "payload_sha256": freeze.load_hashed(plan_path)["payload_sha256"],
    }
    monkeypatch.setattr(freeze, "PREDECESSOR_PLAN", predecessor_plan)
    shutil.copyfile(HERE / "heldout_cache_delivery_manifest.json", delivery_path)
    predecessor_delivery = freeze.identity(delivery_path) | {
        "commit": freeze.PREDECESSOR_DELIVERY["commit"],
        "payload_sha256": freeze.load_hashed(delivery_path)["payload_sha256"],
    }
    monkeypatch.setattr(freeze, "PREDECESSOR_DELIVERY", predecessor_delivery)

    freeze.freeze_plan(
        argparse.Namespace(
            plan=plan_path,
            stager=HERE / "stage_heldout_for_scoring.py",
            renderer=HERE / "render_real_panels.py",
            public_tooling_commit=TOOLING_COMMIT,
        )
    )
    plan = freeze.load_hashed(plan_path)
    record = plan["result_blind_real_panel_index_provenance_correction"]
    assert record == freeze.correction_record(
        TOOLING_COMMIT, freeze.identity(HERE / "render_real_panels.py")
    )
    assert plan["one_shot_scoring_stager"] == freeze.identity(
        HERE / "stage_heldout_for_scoring.py"
    )
    assert plan["real_panel_renderer"] == freeze.identity(
        HERE / "render_real_panels.py"
    )
    assert record["scientific_gate"] == {
        "cache_or_manifest_changed": False,
        "model_metric_threshold_seed_panel_endpoint_gate_aggregation_or_claim_changed": False,
        "test_time_tuning_permitted": False,
        "scientific_outputs_inspected": False,
    }

    freeze.freeze_delivery(
        argparse.Namespace(
            delivery=delivery_path,
            alias=alias_path,
            plan=plan_path,
            public_tooling_commit=TOOLING_COMMIT,
            public_plan_commit=PLAN_COMMIT,
        )
    )
    delivery = freeze.load_hashed(delivery_path)
    assert delivery["public_execution_plan"] == freeze.identity(plan_path) | {
        "commit": PLAN_COMMIT,
        "payload_sha256": plan["payload_sha256"],
    }
    assert delivery["result_blind_real_panel_index_provenance_correction"] == {
        "public_execution_plan_commit": PLAN_COMMIT,
        "public_tooling_commit": TOOLING_COMMIT,
        "exact_cache_job_provenance_required": True,
        "scientific_contract_changed": False,
        "scientific_result_created_or_opened": False,
        "retry_permitted_after_full_preflight": True,
    }
    assert alias_path.read_bytes() == delivery_path.read_bytes()


def test_scientific_projection_allows_only_the_exact_renderer_correction() -> None:
    predecessor = freeze.PREDECESSOR_RENDERER
    corrected_renderer = freeze.identity(HERE / "render_real_panels.py")
    base = {
        "status": "held-out execution plan frozen before held-out inference",
        "real_panel_renderer": predecessor,
        "threshold_binding": {"payload_sha256": "1" * 64},
    }
    corrected = dict(base)
    corrected["real_panel_renderer"] = corrected_renderer
    corrected["result_blind_real_panel_index_provenance_correction"] = (
        freeze.correction_record(TOOLING_COMMIT, corrected_renderer)
    )
    assert stager.scientific_projection(corrected) == stager.scientific_projection(base)

    tampered = dict(corrected)
    tampered["real_panel_renderer"] = dict(corrected_renderer) | {"sha256": "2" * 64}
    with pytest.raises(RuntimeError, match="correction renderer mismatch"):
        stager.scientific_projection(tampered)
