from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "upload_runpod_primary_dataset", HERE / "upload_runpod_primary_dataset.py"
)
uploader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(uploader)


def test_parse_ledger_accepts_canonical_nested_records(tmp_path: Path) -> None:
    ledger = tmp_path / "DATASET_SHA256SUMS"
    ledger.write_text(
        f"{'a' * 64}  primary/job/heldout_job_index.json\n"
        f"{'b' * 64}  primary/job/run/cache.npz\n",
        encoding="utf-8",
    )
    assert uploader.parse_ledger(ledger) == {
        "primary/job/heldout_job_index.json": "a" * 64,
        "primary/job/run/cache.npz": "b" * 64,
    }


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../b", "./file"])
def test_safe_relative_rejects_escape_paths(relative: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        uploader.safe_relative(relative)


def test_uploader_is_result_blind_and_uses_nested_transport() -> None:
    source = (HERE / "upload_runpod_primary_dataset.py").read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "numpy" not in source
    assert "zipfile" not in source
    assert "MAX_FILES_TO_UPLOAD = 1000" in source
    assert "dataset_upload(" in source
