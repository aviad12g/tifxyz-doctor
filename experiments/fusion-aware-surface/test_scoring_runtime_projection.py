from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_runtime_corrections_are_explicitly_operational_plan_fields() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text(encoding="utf-8")
    assert '"result_blind_scoring_runtime_transport_correction"' in source
    assert '"result_blind_scoring_runtime_projection_correction"' in source
    assert '"result_blind_parallel_real_scoring_retry"' in source
    assert '"result_blind_parallel_real_asset_transport"' in source
    assert '"result_blind_verified_parallel_scorer_projection_correction"' in source
    assert 'projected_scorers["real"]["script"] = predecessor_scorer' in source


def test_staging_failure_emits_only_captured_operational_stderr() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    assert "sys.stderr.buffer.write(result.stderr)" in source
    assert 'raise RuntimeError("held-out scoring input staging failed")' in source
