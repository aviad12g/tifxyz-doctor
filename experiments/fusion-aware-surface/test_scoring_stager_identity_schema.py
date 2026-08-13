from __future__ import annotations

import pytest

import stage_heldout_for_scoring as stager


def test_scoring_stager_accepts_frozen_real_cache_identity_schema() -> None:
    stager.validate_cache_identity_record(
        {
            "file": "patch.npz",
            "bytes": 123,
            "sha256": "a" * 64,
            "source_image": "scroll/patch.tif",
        },
        "real_test_cache",
        "real-test-baseline",
    )


def test_scoring_stager_accepts_frozen_synthetic_cache_identity_schema() -> None:
    stager.validate_cache_identity_record(
        {
            "file": "cell.npz",
            "bytes": 456,
            "sha256": "b" * 64,
            "name": "cell",
            "kind": "primary",
            "seed": 11,
            "pitch_um": 7.91,
            "papyrus": 8,
            "kollesis": False,
        },
        "synthetic_ray_cache",
        "synthetic-control-seed11-all-shards",
    )


def test_scoring_stager_rejects_reduced_synthetic_identity_schema() -> None:
    with pytest.raises(RuntimeError, match="cache identity schema mismatch"):
        stager.validate_cache_identity_record(
            {"file": "cell.npz", "bytes": 456, "sha256": "b" * 64},
            "synthetic_ray_cache",
            "synthetic-control-seed11-all-shards",
        )
