from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "adapt_runpod_scoring_transport", HERE / "adapt_runpod_scoring_transport.py"
)
adapter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter)


def test_metadata_adapter_removes_secondary_kernels_and_adds_runpod_dataset() -> None:
    threshold = "owner/threshold/6"
    secondary = [f"owner/secondary-{index}/1" for index in range(7)]
    metadata = {
        "dataset_sources": ["owner/assets/1"],
        "kernel_sources": [threshold, *secondary],
    }
    adapted = adapter.adapt_metadata(
        metadata,
        threshold_kernel=threshold,
        expected_synthetic_kernel_sources=secondary,
        dataset_source="owner/runpod-primary/1",
    )
    assert adapted["kernel_sources"] == [threshold]
    assert adapted["dataset_sources"] == ["owner/assets/1", "owner/runpod-primary/1"]
    assert metadata["kernel_sources"] == [threshold, *secondary]


def test_metadata_adapter_rejects_unexpected_source_order() -> None:
    with pytest.raises(RuntimeError, match="source order mismatch"):
        adapter.adapt_metadata(
            {"dataset_sources": [], "kernel_sources": ["threshold", "wrong"]},
            threshold_kernel="threshold",
            expected_synthetic_kernel_sources=["expected"],
            dataset_source="owner/runpod-primary/1",
        )


def test_rewritten_ledger_is_self_consistent(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("one\n")
    (tmp_path / "kernel-metadata.json").write_text(json.dumps({"x": 1}) + "\n")
    (tmp_path / "KERNEL_SHA256SUMS").write_text("stale\n")
    identities = adapter.rewrite_package_ledger(tmp_path)
    expected = "".join(
        f"{identities[name]['sha256']}  {name}\n"
        for name in sorted(identities)
        if name != "KERNEL_SHA256SUMS"
    )
    assert (tmp_path / "KERNEL_SHA256SUMS").read_text() == expected
    assert identities["KERNEL_SHA256SUMS"]["sha256"] == adapter.sha256_file(
        tmp_path / "KERNEL_SHA256SUMS"
    )


def test_transport_adapter_does_not_read_scientific_results() -> None:
    source = (HERE / "adapt_runpod_scoring_transport.py").read_text()
    assert "np.load" not in source
    assert "numpy" not in source
    assert "sealed_" not in source
