from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import SurfaceGrid, sample_trilinear_zyx  # noqa: E402
from preflight_debug_m7 import (  # noqa: E402
    ChunkGroupedSampler,
    derive_geometry_masks,
    dilate_one_cell,
    evaluate_transects,
    exact_max_support_rectangle,
    largest_area_clean_rectangle,
    verdict_exit_code,
)
from tifxyz_render_pipeline import OpenedVolume  # noqa: E402


class RecordingVolume:
    def __init__(self, data: np.ndarray, chunks: tuple[int, int, int]) -> None:
        self.data = data
        self.shape = data.shape
        self.ndim = 3
        self.dtype = data.dtype
        self.chunks = chunks
        self.read_count = 0

    def __getitem__(self, key: object) -> np.ndarray:
        self.read_count += 1
        return self.data[key]


def brute_best(
    forbidden: np.ndarray,
    area: np.ndarray,
    supported: np.ndarray,
    valid: np.ndarray,
    low: float,
    high: float,
) -> dict[str, object] | None:
    height, width = forbidden.shape
    best = None
    for row0, row1, column0, column1 in itertools.product(
        range(height - 1), range(2, height + 1), range(width - 1), range(2, width + 1)
    ):
        if row1 - row0 < 2 or column1 - column0 < 2:
            continue
        if forbidden[row0:row1, column0:column1].any():
            continue
        area_by_column = area[row0 : row1 - 1].sum(axis=0, dtype=np.float64)
        area_prefix = np.pad(np.cumsum(area_by_column, dtype=np.float64), (1, 0))
        candidate_area = float(area_prefix[column1 - 1] - area_prefix[column0])
        if not low <= candidate_area <= high:
            continue
        support_count = int(supported[row0:row1, column0:column1].sum())
        valid_count = int(valid[row0:row1, column0:column1].sum())
        if not valid_count:
            continue
        box = (row0, row1, column0, column1)
        candidate = {
            "box": box,
            "area": candidate_area,
            "support": support_count,
            "valid": valid_count,
        }
        if best is None:
            best = candidate
            continue
        left = support_count * int(best["valid"])
        right = int(best["support"]) * valid_count
        if left > right or (
            left == right
            and (
                candidate_area > float(best["area"])
                or (candidate_area == float(best["area"]) and box < best["box"])
            )
        ):
            best = candidate
    return best


def test_one_cell_dilation_is_eight_connected_and_clipped() -> None:
    source = np.zeros((5, 6), dtype=bool)
    source[0, 0] = True
    source[3, 4] = True
    result = dilate_one_cell(source)
    expected = np.zeros_like(source)
    expected[0:2, 0:2] = True
    expected[2:5, 3:6] = True
    np.testing.assert_array_equal(result, expected)


def test_geometry_mask_marks_offset_out_and_fold_margin() -> None:
    rows, columns = np.mgrid[:5, :6]
    surface = SurfaceGrid.from_tifxyz(
        columns.astype(float) + 10.0,
        rows.astype(float) + 10.0,
        np.full((5, 6), 20.0),
        mask=np.ones((5, 6), dtype=bool),
    )
    masks = derive_geometry_masks(
        surface, volume_shape_zyx=(50, 50, 50), voxel_um=1.0, maximum_offset_voxels=2
    )
    assert not masks.forbidden.any()
    assert masks.summary["raw_reason_vertex_counts"]["folded_quad"] == 0
    assert masks.quad_area_cm2.shape == (4, 5)


def test_geometry_mask_accepts_explicit_subset_mask_but_never_invalid_coordinate() -> (
    None
):
    rows, columns = np.mgrid[:7, :7]
    valid = np.ones((7, 7), dtype=bool)
    valid[3, 3] = False
    surface = SurfaceGrid.from_tifxyz(
        columns.astype(float) + 10.0,
        rows.astype(float) + 10.0,
        np.full((7, 7), 20.0),
        mask=valid,
    )
    masks = derive_geometry_masks(
        surface, volume_shape_zyx=(50, 50, 50), voxel_um=1.0, maximum_offset_voxels=2
    )
    assert not masks.summary["strict_validity_matches_loader"]
    assert masks.summary["explicitly_masked_finite_nonnegative_coordinate_count"] == 1
    invalid_coordinate = SurfaceGrid.from_tifxyz(
        np.where(valid, columns + 10.0, -1.0),
        rows + 10.0,
        np.full((7, 7), 20.0),
        mask=np.ones((7, 7), dtype=bool),
    )
    with np.testing.assert_raises_regex(ValueError, "includes a nonfinite or negative"):
        derive_geometry_masks(
            invalid_coordinate,
            volume_shape_zyx=(50, 50, 50),
            voxel_um=1.0,
            maximum_offset_voxels=2,
        )


def test_area_search_uses_two_triangle_quads_and_inclusive_bounds() -> None:
    forbidden = np.zeros((4, 5), dtype=bool)
    area = np.full((3, 4), 0.125, dtype=np.float64)
    result = largest_area_clean_rectangle(
        forbidden, area, min_area_cm2=0.5, max_area_cm2=0.5
    )
    assert result["feasible_rectangle_count"] > 0
    assert result["largest_feasible_rectangle"]["area_cm2"] == 0.5
    forbidden[1, 2] = True
    restricted = largest_area_clean_rectangle(
        forbidden, area, min_area_cm2=0.5, max_area_cm2=0.5
    )
    box = restricted["largest_feasible_rectangle"]["window_r0_r1_c0_c1"]
    row0, row1, column0, column1 = box
    assert not forbidden[row0:row1, column0:column1].any()


def test_exact_support_optimizer_matches_brute_force_randomized() -> None:
    rng = np.random.default_rng(1447)
    for height, width in ((4, 6), (6, 4), (7, 7)):
        for _ in range(12):
            forbidden = rng.random((height, width)) < 0.12
            valid = ~forbidden
            supported = (rng.random((height, width)) < 0.55) & valid
            area = rng.integers(1, 5, size=(height - 1, width - 1)).astype(float) / 40.0
            low, high = 0.2, 1.1
            expected = brute_best(forbidden, area, supported, valid, low, high)
            actual = exact_max_support_rectangle(
                forbidden,
                area,
                supported,
                valid,
                min_area_cm2=low,
                max_area_cm2=high,
            )["best_rectangle"]
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert tuple(actual["window_r0_r1_c0_c1"]) == expected["box"]
                assert actual["supported_vertex_count"] == expected["support"]
                assert actual["valid_vertex_count"] == expected["valid"]


def test_chunk_grouped_sampler_matches_native_and_crosses_chunk_boundary() -> None:
    volume_data = np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11)
    volume = RecordingVolume(volume_data, chunks=(4, 4, 4))
    opened = OpenedVolume(
        volume_zyx=volume,  # type: ignore[arg-type]
        source="synthetic",
        array_path="0",
        axes="zyx",
        metadata={},
    )
    coordinates = np.array(
        [[3.8, 3.7, 3.9], [4.2, 4.1, 4.3], [7.9, 2.4, 6.8], [10.0, 9.0, 8.0]],
        dtype=np.float64,
    )
    expected = sample_trilinear_zyx(volume_data, coordinates)
    sampler = ChunkGroupedSampler(opened)
    values, valid = sampler.sample(coordinates, point_valid=np.ones(4, dtype=bool))
    np.testing.assert_allclose(values, expected.values, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(valid, expected.valid)
    assert volume.read_count == sampler.manifest()["roi_read_count"]


def test_deterministic_transects_require_one_centered_run() -> None:
    frames = np.zeros((31, 5, 10), dtype=np.uint8)
    frames[13:18] = 255
    result = evaluate_transects(frames, (0, 5, 0, 10), seed=1447)
    assert result["pass"]
    assert result["pass_count"] == 50
    frames[3, 0, 0] = 255
    result_again = evaluate_transects(frames, (0, 5, 0, 10), seed=1447)
    assert (
        result_again["chosen_flat_indices_local"] == result["chosen_flat_indices_local"]
    )
    assert result_again["pass_count"] == 49
    assert result_again["pass"] is False


def test_verdict_exit_codes_stop_on_fail_or_inconclusive() -> None:
    assert verdict_exit_code("preflight_pass") == 0
    assert verdict_exit_code("fail") == 2
    assert verdict_exit_code("inconclusive") == 3
