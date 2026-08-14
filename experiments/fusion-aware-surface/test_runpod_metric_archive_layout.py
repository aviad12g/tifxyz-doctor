"""Checks for the exact public metric archive wrapper layout."""

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_wrapper_accepts_only_exact_source_and_bundled_wheels_layout() -> None:
    source = (HERE / "runpod_prepare_and_execute_from_kaggle.py").read_text(
        encoding="utf-8"
    )
    assert 'bundled_wheels = staging / "wheels"' in source
    assert "set(staging.iterdir()) != {extracted, bundled_wheels}" in source
    assert "shutil.rmtree(bundled_wheels)" in source
    assert 'extracted = staging / "topological-metrics-kaggle"' in source
