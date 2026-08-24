from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_launcher_materializes_exact_public_scorer_dependencies() -> None:
    source = (HERE / "one_shot_scoring_launcher.py").read_text(encoding="utf-8")
    for name in (
        "fusion_loss.py",
        "gap_supervision.py",
        "inference.py",
        "normalization.py",
        "train_fusion_aware.py",
    ):
        assert f'"{name}"' in source
    assert "materialize_public_scorer_imports(" in source
    assert "fetch_public(commit, name, PROJECT_HASHES[name])" in source
    assert '"scorer_public_imports": scorer_imports' in source


def test_public_import_freezer_preserves_result_blind_scope() -> None:
    source = (HERE / "freeze_scorer_public_import_dependencies.py").read_text(
        encoding="utf-8"
    )
    assert '"dependency_source_bytes_changed": False' in source
    assert '"private_transport_verified": True' in source
    assert '"scientific_outputs_inspected": False' in source
    assert '"npz_panel_probability_endpoint_or_result_opened": False' in source
