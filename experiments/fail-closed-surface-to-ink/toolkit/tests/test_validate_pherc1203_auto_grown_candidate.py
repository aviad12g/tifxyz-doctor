from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import SurfaceGrid  # noqa: E402
from sample_raw_ct_seed_cube import ZarrV2ArraySpec  # noqa: E402
from validate_pherc0800_masked_candidate import (  # noqa: E402
    conservative_manifold_subset,
)
from validate_pherc1203_auto_grown_candidate import (  # noqa: E402
    derived_full_shape,
    evaluate_stored_surface,
    stored_gate_results,
)


class CenteredSlabSampler:
    def sample(self, coordinates, *, point_valid):
        points = np.asarray(coordinates)
        values = (np.abs(points[..., 2] - 300.0) <= 2.0).astype(np.uint8)
        valid = np.asarray(point_valid, dtype=bool)
        return values, valid


def test_derived_full_shape_snaps_official_float32_scale() -> None:
    assert derived_full_shape((147, 141), (0.05000000074505806,) * 2) == (
        2940,
        2820,
    )


def test_radius46_both_sign_evaluation_is_exact_reversal_and_gates() -> None:
    rows, columns = np.mgrid[:5, :6]
    surface = SurfaceGrid.from_tifxyz(
        (columns + 100).astype(np.float32),
        (rows + 200).astype(np.float32),
        np.full((5, 6), 300, dtype=np.float32),
        mask=np.ones((5, 6), dtype=bool),
    )
    spec = ZarrV2ArraySpec(
        shape_zyx=(1000, 1000, 1000),
        chunks_zyx=(64, 64, 64),
        dtype_string="|u1",
        order="C",
        fill_value=0,
        dimension_separator=".",
        compressor=None,
        filters=None,
    )
    result = evaluate_stored_surface(surface, CenteredSlabSampler(), spec)
    assert result["geometry"]["pass"]
    assert all(result["sign_reversal_invariants"].values())
    assert result["orientations"]["positive"]["frames"].shape == (93, 5, 6)
    assert result["orientations"]["negative"]["sampling_complete"]

    repaired, repair = conservative_manifold_subset(
        result["largest_mask"], result["masks"].quad_area_cm2
    )
    thresholds = {
        "plus_minus2_support_fraction": 0.9875,
        "offset0_support_fraction": 0.8625,
        "continuity_center_jump_p95": 2.0,
        "continuity_interval_overlap_fraction": 1.0,
        "competitor_clearance_minimum": 1.0,
        "competitor_clearance_p05": 4.0,
        "competitor_clearance_median": 7.0,
    }
    gates = stored_gate_results(
        result, repair, repaired, thresholds, minimum_area_cm2=0.0
    )
    assert all(gate["pass"] for gate in gates)
