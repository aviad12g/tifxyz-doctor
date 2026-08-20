import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FREEZE = load_module("freeze_gapbalance_runpod_development", "freeze_gapbalance_runpod_development.py")
WRAPPER = load_module("run_gapbalance_runpod_development_job", "run_gapbalance_runpod_development_job.py")


def test_public_plan_has_exact_cap_waves_and_blind_gates():
    plan = WRAPPER.load_plan(HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json")
    assert plan["budget"] == {
        "absolute_campaign_cap_usd": 25.0,
        "authorization_date": "2026-08-20",
        "confirmation_reserve_usd": 13.0,
        "development_billing_cutoff_usd": 12.0,
        "forms_authorized": False,
        "guard_seconds": 300,
        "on_cutoff": "stop the exact pod, preserve partial operational artifacts, do not score or select partial development output",
        "user_authorization": "Aviad: ok go",
    }
    assert [len(wave) for wave in plan["execution"]["waves"]] == [7, 5]
    assert [job for wave in plan["execution"]["waves"] for job in wave] == [
        job["job_id"] for job in plan["jobs"]
    ]
    assert len(plan["jobs"]) == 12
    assert plan["authority"]["partial_primary_scoring_or_selection_permitted"] is False
    assert all(value is False for value in plan["sealed_gates"].values())


def test_plan_hash_rejects_budget_or_authority_mutation(tmp_path):
    path = HERE / "GAPBALANCE_RUNPOD_DEVELOPMENT_PLAN.json"
    plan = json.loads(path.read_text())
    altered = copy.deepcopy(plan)
    altered["budget"]["development_billing_cutoff_usd"] = 25.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="payload mismatch"):
        WRAPPER.load_plan(changed)
    altered = copy.deepcopy(plan)
    altered["authority"]["partial_primary_scoring_or_selection_permitted"] = True
    changed.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="payload mismatch"):
        WRAPPER.load_plan(changed)


def test_import_fix_supplies_exact_12_synthetic_jobs():
    fix = FREEZE.load_hashed(HERE / "GAPBALANCE_DEVELOPMENT_IMPORT_FIX.json")
    jobs = [job for job in fix["jobs"] if job["job_id"].startswith("gapbalance-development-synthetic-")]
    assert len(jobs) == 12
    assert [job["job_id"] for job in jobs] == [
        f"gapbalance-development-synthetic-seed{seed}-shard{shard:02d}"
        for seed in (11, 23, 47)
        for shard in range(4)
    ]
