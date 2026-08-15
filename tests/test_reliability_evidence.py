from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_reliability_evidence.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_reliability_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_reliability_evidence_verifies() -> None:
    result = _load_verifier().verify()
    assert result["status"] == "verified"
    assert result["current_pull_request"] == 1299
    assert result["local_artifacts_verified"] == 9
    assert result["historical_references"] == 3
