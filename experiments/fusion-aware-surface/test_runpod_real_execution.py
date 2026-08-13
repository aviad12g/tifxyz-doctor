import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collector_download_is_npz_only_and_result_blind() -> None:
    source = (HERE / "collect_runpod_real_scoring_inputs.py").read_text(encoding="utf-8")
    assert '"--page-size",\n            "100"' in source
    assert '"--page-token",\n            ""' in source
    assert '"npz_payloads_opened_or_parsed": False' in source
    assert "5_791_045_122" in source


def test_executor_preserves_sealed_outputs_and_runs_fixed_launcher() -> None:
    source = (HERE / "runpod_execute_real_scoring.py").read_text(encoding="utf-8")
    assert '"scientific_outputs_inspected": False' in source
    assert '"KAGGLE_INPUT_PATH": "/workspace"' in source
    assert '"sealed_real_test_results.json"' in source
    assert '"REAL_SCORING_COMPLETE"' in source


def test_orchestrator_has_manual_wall_clock_budget_enforcement() -> None:
    module = load("runpod_real_orchestrator", "orchestrate_runpod_real_scoring.py")
    source = (HERE / "orchestrate_runpod_real_scoring.py").read_text(encoding="utf-8")
    assert 'guard_seconds=receipt["guard_seconds"]' in source
    assert 'guarded >= receipt["compute_cutoff_usd"]' in source
    assert "runpod.stop_pod" in source
    assert callable(module.spend)


def test_deployer_uses_guarded_rsync_and_absent_remote_targets() -> None:
    source = (HERE / "deploy_runpod_real_scoring.py").read_text(encoding="utf-8")
    for flag in ("--archive", "--copy-links", "--partial", "--protect-args"):
        assert f'"{flag}"' in source
    assert "test ! -e /workspace/real-scoring-status" in source
    assert "test ! -e /workspace/real-scoring-input" in source


def test_execution_freezer_binds_exact_budget_and_private_scope() -> None:
    source = (HERE / "freeze_runpod_real_execution.py").read_text(encoding="utf-8")
    assert '"absolute_cap_usd": 3.50' in source
    assert '"compute_cutoff_usd": 3.25' in source
    assert '"private_npz_payloads_opened_before_transfer": False' in source
    assert '"new_private_checkpoint_or_research_input_upload": False' in source
