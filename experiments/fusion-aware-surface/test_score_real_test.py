import json

import numpy as np
import pytest

from score_real_test import (
    bootstrap_mean_interval,
    canonical_sha256,
    validate_embedded_hash,
)


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean() -> None:
    values = np.array([-0.02, 0.01, 0.03, 0.04], dtype=np.float64)
    first = bootstrap_mean_interval(values, seed=7, replicates=1000)
    second = bootstrap_mean_interval(values, seed=7, replicates=1000)
    assert first == second
    assert first["mean"] == pytest.approx(values.mean())
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]


def test_embedded_hash_validation_fails_closed(tmp_path) -> None:
    payload = {"status": "frozen", "runs": {}}
    payload["payload_sha256"] = canonical_sha256(payload)
    validate_embedded_hash(payload)
    broken = json.loads(json.dumps(payload))
    broken["status"] = "changed"
    with pytest.raises(RuntimeError, match="does not match"):
        validate_embedded_hash(broken)
