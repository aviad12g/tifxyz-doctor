"""Structural checks for the public-rsync replacement CPU freeze."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_rsync_retry_is_result_blind_cpu_only_and_budgeted() -> None:
    source = (HERE / "freeze_new_runpod_parallel_rsync_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 19.98' in source
    assert '"install_public_rsync_before_transfer": True' in source
    assert '"private_file_transfer_started_in_failed_attempt": False' in source
    assert '"held_out_result_opened_or_used": False' in source
    assert '"parallel_workers": 32' in source
    assert '"gpu_count": 0' in source
