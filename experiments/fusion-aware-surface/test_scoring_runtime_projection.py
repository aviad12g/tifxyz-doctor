from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_runtime_corrections_are_explicitly_operational_plan_fields() -> None:
    source = (HERE / "stage_heldout_for_scoring.py").read_text(encoding="utf-8")
    assert '"result_blind_scoring_runtime_transport_correction"' in source
    assert '"result_blind_scoring_runtime_projection_correction"' in source
