from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_metric_worker_retry_is_cpu_only_and_result_blind() -> None:
    source = (HERE / "freeze_new_runpod_metric_worker_retry.py").read_text(encoding="utf-8")
    assert '"gpu_count": 0' in source
    assert '"parallel_workers": 32' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"private_signed_bundle_transport_verified": True' in source
    assert '"metric_worker_source_bytes_changed": False' in source
    assert '"scorer_default_worker_path_changed": False' in source
    assert '"npz_panel_probability_endpoint_or_result_opened": False' in source
