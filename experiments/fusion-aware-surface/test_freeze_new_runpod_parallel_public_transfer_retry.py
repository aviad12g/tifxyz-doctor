from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_parallel_public_transfer_freezer_is_result_blind_and_bounded() -> None:
    source = (HERE / "freeze_new_runpod_parallel_public_transfer_retry.py").read_text(
        encoding="utf-8"
    )
    assert '"compute_cutoff_usd": 14.0' in source
    assert '"maximum_parallel_ssh_streams": 8' in source
    assert '"metric_runtime_tree_identity_changed": False' in source
    assert '"parallel_scorer_workers": 32' in source
    assert '"gpu_count": 0' in source
