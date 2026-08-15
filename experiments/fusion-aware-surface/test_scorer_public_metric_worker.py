from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_launcher_materializes_exact_public_metric_worker_sibling() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    assert "def materialize_public_metric_worker(" in source
    assert 'name = "official_metric.py"' in source
    assert "fetch_public(commit, name, PROJECT_HASHES[name])" in source
    assert '"scorer_public_metric_worker": scorer_metric_worker' in source


def test_metric_worker_freezer_is_result_blind() -> None:
    source = (HERE / "freeze_scorer_public_metric_worker.py").read_text(encoding="utf-8")
    assert '"metric_worker_source_bytes_changed": False' in source
    assert '"scorer_default_worker_path_changed": False' in source
    assert '"private_transport_verified": True' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"npz_panel_probability_endpoint_or_result_opened": False' in source
