"""Structural checks for the public metric archive layout retry."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_archive_layout_retry_is_public_only_and_result_blind() -> None:
    source = (HERE / "freeze_new_runpod_metric_archive_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 19.75' in source
    assert '"public metric archive layout mismatch"' in source
    assert '"topological-metrics-kaggle"' in source
    assert '"wheels"' in source
    assert '"private_dataset_pull_started": False' in source
    assert '"real_scorer_started": False' in source
    assert '"metric_source_bytes_changed": False' in source
