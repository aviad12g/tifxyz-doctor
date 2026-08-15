"""Structural checks for the corrected executor contract retry."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_executor_contract_retry_is_result_blind_and_budgeted() -> None:
    source = (HERE / "freeze_new_runpod_execution_contract_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 17.5' in source
    assert '"reserve_usd": 1.0' in source
    assert '"old_executor_guaranteed_to_reject_before_science": True' in source
    assert '"exact_cpu_provider_contract_required": True' in source
    assert '"exact_32_worker_equivalence_contract_required": True' in source
    assert '"all_existing_file_and_runtime_identity_checks_retained": True' in source
    assert '"npz_panel_probability_endpoint_or_result_opened": False' in source
    assert '"gpu_count": 0' in source
