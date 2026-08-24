from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from select_m7_seed_chunk import (  # noqa: E402
    SelectorConfig,
    contiguous_true_runs,
    estimate_pca_normal,
    evaluate_candidate,
    evaluate_normal_transect,
    select_candidate,
)


def planar_mask(
    *, z: int = 96, y0: int = 60, y1: int = 133, x0: int = 60, x1: int = 133
) -> np.ndarray:
    mask = np.zeros((192, 192, 192), dtype=bool)
    mask[z, y0:y1, x0:x1] = True
    return mask


def test_contiguous_true_runs_are_half_open() -> None:
    line = np.array([False, True, True, False, True, False], dtype=bool)
    assert contiguous_true_runs(line) == ((1, 3), (4, 5))


def test_pca_normal_is_deterministically_oriented_for_flat_sheet() -> None:
    mask = planar_mask()
    config = SelectorConfig()
    estimate, reason = estimate_pca_normal(mask, (96, 96, 96), config)
    assert reason == "pass"
    assert estimate is not None
    np.testing.assert_allclose(estimate.normal_zyx, (1.0, 0.0, 0.0), atol=1e-12)
    assert estimate.normal_to_tangent_variance_ratio == 0.0


def test_transect_rejects_a_competing_parallel_sheet() -> None:
    mask = planar_mask()
    mask[104, 60:133, 60:133] = True
    config = SelectorConfig()
    result = evaluate_normal_transect(mask, (96, 96, 96), (1, 0, 0), config)
    assert not result.passed
    assert result.reason == "competing_foreground_run"
    assert result.runs_inclusive_offsets == ((0, 0), (8, 8))


def test_candidate_requires_all_four_tangential_probes() -> None:
    mask = planar_mask()
    config = SelectorConfig()
    passing = evaluate_candidate(mask, (96, 96, 96), config)
    assert passing["pass"]
    assert len(passing["tangential_probes"]) == 4
    assert all(probe["pass"] for probe in passing["tangential_probes"])

    # Remove every possible snap target around one t1 direction.  The center
    # PCA neighborhood remains intact, so rejection is specifically a probe.
    mask[96, 92:101, 101:112] = False
    rejected = evaluate_candidate(mask, (96, 96, 96), config)
    assert not rejected["pass"]
    assert "no_nearby_foreground" in rejected["reason"]


def test_selector_skips_nearest_ambiguous_layer_and_is_deterministic() -> None:
    mask = planar_mask(z=96, y0=50, y1=143, x0=50, x1=143)
    # A compact second sheet makes the anchor's local PCA/transect ambiguous.
    # A more distant area on the main plane remains clean.
    mask[101, 90:103, 90:103] = True
    chunk = mask.astype(np.uint8) * 255
    config = SelectorConfig(search_radius=16)
    first = select_candidate(chunk, (96, 96, 96), config)
    second = select_candidate(chunk, (96, 96, 96), config)
    assert first["pass"]
    assert second["pass"]
    assert first["selected"]["local_zyx"] == second["selected"]["local_zyx"]
    assert first["selected"]["local_zyx"] != [96, 96, 96]
    assert first["evaluated_candidate_count"] > 1


def test_selector_rejects_search_that_crosses_chunk_boundary() -> None:
    chunk = np.zeros((192, 192, 192), dtype=np.uint8)
    config = SelectorConfig(search_radius=32)
    try:
        select_candidate(chunk, (96, 170, 96), config)
    except ValueError as error:
        assert "chunk edge" in str(error)
    else:
        raise AssertionError("expected chunk-edge guard")
