"""Structural checks for the fresh verified-parallel CPU retry freezer."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_fresh_retry_freezer_binds_cpu_parallel_and_budget_contracts() -> None:
    source = (HERE / "freeze_new_runpod_parallel_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"absolute_cap_usd": 21.0' in source
    assert '"compute_cutoff_usd": 20.0' in source
    assert '"reserve_usd": 1.0' in source
    assert '"parallel_workers": 32' in source
    assert '"gpu_count": 0' in source
    assert '"private_dataset_version": 1' in source
    assert '"held_out_result_opened_or_used": False' in source
