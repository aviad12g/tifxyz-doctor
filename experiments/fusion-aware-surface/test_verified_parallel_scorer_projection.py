import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent


def load_stager():
    spec = importlib.util.spec_from_file_location(
        "verified_parallel_projection_stager", HERE / "stage_heldout_for_scoring.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_plan_projects_to_exact_cache_job_science() -> None:
    stager = load_stager()
    current = json.loads((HERE / "heldout_execution_plan.json").read_text(encoding="utf-8"))
    current["result_blind_verified_parallel_scorer_projection_correction"] = {
        "result_blind": True
    }
    predecessor = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                "46d030245c14ecb2be63796a308581c462fc2e29:experiments/fusion-aware-surface/heldout_execution_plan.json",
            ]
        )
    )
    assert stager.scientific_projection(current) == stager.scientific_projection(
        predecessor
    )


def test_projection_freezer_records_result_blind_scope() -> None:
    source = (HERE / "freeze_verified_parallel_scorer_projection.py").read_text(
        encoding="utf-8"
    )
    assert '"projection_scope": "one_shot_scorers.real.script"' in source
    assert '"held_out_result_opened_or_used": False' in source
    assert '"scientific_contract_changed": False' in source
