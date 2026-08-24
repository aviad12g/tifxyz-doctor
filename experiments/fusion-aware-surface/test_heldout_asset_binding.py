from __future__ import annotations

import hashlib
from pathlib import Path

import heldout_cache_launcher as launcher
import pytest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "project" / "source.py"
    source.parent.mkdir()
    source.write_bytes(b"ledger-bound source\n")
    checkpoint = tmp_path / "model" / "Model_epoch499.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"directly hash-bound checkpoint\n")
    ledger = tmp_path / "SOURCE_SHA256SUMS"
    ledger.write_text(f"{_sha256(source.read_bytes())}  project/source.py\n")

    monkeypatch.setattr(launcher, "ASSET_LEDGER_SHA256", launcher.sha256_file(ledger))
    monkeypatch.setattr(launcher, "ASSET_LEDGER_RECORDS", 1)
    monkeypatch.setattr(
        launcher, "SOURCE_HASHES", {"project/source.py": launcher.sha256_file(source)}
    )
    monkeypatch.setattr(
        launcher,
        "DIRECT_SOURCE_IDENTITIES",
        {
            "model/Model_epoch499.pth": (
                checkpoint.stat().st_size,
                launcher.sha256_file(checkpoint),
            )
        },
    )
    return checkpoint


def test_checkpoint_can_be_directly_bound_outside_source_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    launcher.verify_asset_root(tmp_path)


def test_direct_checkpoint_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _fixture(tmp_path, monkeypatch)
    checkpoint.write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="direct identity mismatch"):
        launcher.verify_asset_root(tmp_path)
