import json

import pytest

from cache_real_predictions import require_test_gate, run_identity


def test_run_identity_contract() -> None:
    assert run_identity("baseline") == (None, None)
    assert run_identity("control_seed11") == ("control", 11)
    assert run_identity("gap8_seed47") == ("gap8", 47)
    with pytest.raises(ValueError):
        run_identity("gap8_seed99")


def test_test_gate_requires_matching_frozen_manifest(tmp_path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {
                "status": "thresholds frozen from Scroll-1 validation before test inference",
                "source_split_records_sha256": "abc",
                "runs": {"baseline": {"selected_threshold": 0.4}},
            }
        )
    )
    assert require_test_gate(path, "baseline", "abc") == 0.4
    with pytest.raises(RuntimeError, match="different real-data split"):
        require_test_gate(path, "baseline", "def")
    with pytest.raises(RuntimeError, match="requires"):
        require_test_gate(None, "baseline", "abc")
