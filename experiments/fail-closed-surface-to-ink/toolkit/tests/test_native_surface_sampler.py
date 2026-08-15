from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


# ``first-letters`` is deliberately a script directory rather than an installed
# Python package (its hyphen is not importable).  Import the module under test
# directly without reaching into the vendored villa tree.
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import (  # noqa: E402
    NativeSurfaceSampler,
    NormalSamplingSpec,
    SurfaceGrid,
    estimate_surface_normals,
    sample_trilinear_zyx,
    validate_surface_geometry,
)


def make_xyz_plane(
    height: int,
    width: int,
    *,
    z: float,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.mgrid[:height, :width]
    x = columns.astype(np.float64) + x_origin
    y = rows.astype(np.float64) + y_origin
    z_array = np.full((height, width), z, dtype=np.float64)
    return x, y, z_array


def linear_ramp_volume(shape_zyx: tuple[int, int, int]) -> np.ndarray:
    z, y, x = np.indices(shape_zyx, dtype=np.float64)
    return 100.0 * z + 10.0 * y + x


def test_trilinear_sampling_uses_volume_zyx_and_coordinates_xyz() -> None:
    volume = linear_ramp_volume((6, 7, 8))
    coordinates_xyz = np.asarray(
        [
            [1.25, 2.5, 3.75],
            [7.0, 6.0, 5.0],  # exact final voxel center remains valid
        ]
    )

    sampled = sample_trilinear_zyx(volume, coordinates_xyz, output_dtype=np.float64)

    expected = np.asarray(
        [100.0 * 3.75 + 10.0 * 2.5 + 1.25, 100.0 * 5 + 10.0 * 6 + 7]
    )
    np.testing.assert_allclose(sampled.values, expected, rtol=0.0, atol=1e-12)
    assert sampled.valid.tolist() == [True, True]


def test_trilinear_masks_out_points_holes_and_nonfinite_source_voxels() -> None:
    volume = linear_ramp_volume((4, 4, 4))
    volume[1, 1, 1] = np.nan
    volume_mask = np.ones(volume.shape, dtype=bool)
    volume_mask[2, 2, 2] = False
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [-0.01, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    point_valid = np.asarray([False, True, True, True, True])

    sampled = sample_trilinear_zyx(
        volume,
        coordinates,
        point_valid=point_valid,
        volume_mask_zyx=volume_mask,
        fill_value=-99.0,
    )

    assert not sampled.valid.any()
    np.testing.assert_array_equal(sampled.values, np.full(5, -99.0, dtype=np.float32))


def test_normals_have_explicit_positive_and_negative_orientation() -> None:
    x, y, z = make_xyz_plane(5, 6, z=3.0)
    surface = SurfaceGrid.from_tifxyz(x, y, z)

    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(2.0, 1.5, 1.0))

    assert normals.valid.all()
    expected_positive = np.zeros((5, 6, 3), dtype=np.float64)
    expected_positive[..., 2] = 1.0
    np.testing.assert_allclose(normals.positive_xyz, expected_positive, atol=0.0)
    np.testing.assert_allclose(normals.negative_xyz, -expected_positive, atol=0.0)


def test_anisotropic_physical_normal_matches_tilted_plane() -> None:
    rows, columns = np.mgrid[:5, :6]
    x = columns.astype(np.float64)
    y = rows.astype(np.float64)
    # With (z,y,x) voxel sizes (2,1,1), physical z = 2*x_voxels.
    z = 0.5 * x + 4.0
    surface = SurfaceGrid.from_tifxyz(x, y, z)

    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(2.0, 1.0, 1.0))

    # Physical plane is z_um = x_um + 8, so +z-oriented normal is (-1,0,1)/sqrt(2).
    expected = np.asarray([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(
        normals.positive_xyz,
        np.broadcast_to(expected, normals.positive_xyz.shape),
        rtol=0.0,
        atol=1e-12,
    )


def test_default_26_frame_stack_uses_calibrated_physical_offsets() -> None:
    # Intensity is z index, making the sampled displacement directly observable.
    z_indices = np.arange(120, dtype=np.float64)[:, None, None]
    volume = np.broadcast_to(z_indices, (120, 8, 9)).copy()
    x, y, z = make_xyz_plane(4, 5, z=60.0, x_origin=2.0, y_origin=2.0)
    surface = SurfaceGrid.from_tifxyz(x, y, z)
    sampler = NativeSurfaceSampler(
        volume,
        surface,
        voxel_size_zyx_um=(2.0, 1.0, 1.0),
        output_dtype=np.float64,
    )
    spec = NormalSamplingSpec()  # 26 frames, 8 um physical spacing

    rendered = sampler.render_stack(spec)

    assert rendered.values.shape == (26, 4, 5)
    assert rendered.valid.all()
    expected_z_voxels = 60.0 + spec.offsets_um / 2.0
    np.testing.assert_allclose(
        rendered.values,
        np.broadcast_to(expected_z_voxels[:, None, None], rendered.values.shape),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.diff(rendered.values[:, 0, 0]), 4.0, atol=0.0)


def test_normal_sign_reverses_asymmetric_offset_rendering() -> None:
    z_indices = np.arange(10, dtype=np.float64)[:, None, None]
    volume = np.broadcast_to(z_indices, (10, 6, 6)).copy()
    x, y, z = make_xyz_plane(3, 3, z=4.0, x_origin=1.0, y_origin=1.0)
    sampler = NativeSurfaceSampler(
        volume,
        SurfaceGrid.from_tifxyz(x, y, z),
        voxel_size_zyx_um=(2.0, 1.0, 1.0),
        output_dtype=np.float64,
    )

    positive = sampler.render_stack(NormalSamplingSpec.from_offsets([0.0, 2.0, 4.0]))
    negative = sampler.render_stack(
        NormalSamplingSpec.from_offsets([0.0, 2.0, 4.0], normal_sign=-1)
    )

    np.testing.assert_allclose(positive.values[:, 1, 1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(negative.values[:, 1, 1], [4.0, 3.0, 2.0])


def test_surface_hole_is_never_sampled_and_does_not_wrap_gradients() -> None:
    volume = linear_ramp_volume((10, 10, 10))
    x, y, z = make_xyz_plane(5, 5, z=4.0, x_origin=2.0, y_origin=2.0)
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    # Non-finite tifxyz coordinates are also holes even if the supplied mask says valid.
    x[0, 0] = np.nan
    surface = SurfaceGrid.from_tifxyz(x, y, z, mask)
    sampler = NativeSurfaceSampler(
        volume,
        surface,
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        fill_value=-7.0,
    )
    rendered = sampler.render_stack(NormalSamplingSpec.from_offsets([0.0]))

    assert not rendered.valid[0, 2, 2]
    assert rendered.values[0, 2, 2] == -7.0
    assert not rendered.valid[0, 0, 0]
    assert rendered.values[0, 0, 0] == -7.0
    # Pixels beside the interior hole retain one-sided, local gradients.
    assert rendered.valid[0, 2, 1]
    assert rendered.valid[0, 2, 3]
    np.testing.assert_allclose(sampler.normals.positive_xyz[2, 1], [0.0, 0.0, 1.0])


def test_tiled_render_is_bit_identical_at_every_seam() -> None:
    volume = linear_ramp_volume((24, 22, 25))
    rows, columns = np.mgrid[:11, :13]
    x = columns.astype(np.float64) + 4.1
    y = rows.astype(np.float64) + 3.2
    z = 10.0 + 0.03 * columns**2 + 0.02 * rows**2 + 0.01 * rows * columns
    surface = SurfaceGrid.from_tifxyz(x, y, z)
    sampler = NativeSurfaceSampler(
        volume,
        surface,
        voxel_size_zyx_um=(1.4, 1.1, 0.9),
        output_dtype=np.float64,
    )
    spec = NormalSamplingSpec.from_offsets([-1.7, -0.2, 0.0, 1.3, 2.1])

    full = sampler.render_stack(spec)
    tiled = sampler.render_stack_tiled(tile_shape=(4, 5), spec=spec)

    np.testing.assert_array_equal(tiled.valid, full.valid)
    # Exact equality, including rows/columns that coincide with tile boundaries.
    np.testing.assert_array_equal(tiled.values, full.values)
    for seam_row in (4, 8):
        np.testing.assert_array_equal(
            tiled.values[:, seam_row - 1 : seam_row + 1],
            full.values[:, seam_row - 1 : seam_row + 1],
        )
    for seam_column in (5, 10):
        np.testing.assert_array_equal(
            tiled.values[:, :, seam_column - 1 : seam_column + 1],
            full.values[:, :, seam_column - 1 : seam_column + 1],
        )


def test_geometry_validator_reports_healthy_planar_metrics() -> None:
    x, y, z = make_xyz_plane(5, 6, z=5.0, x_origin=2.0, y_origin=3.0)
    surface = SurfaceGrid.from_tifxyz(x, y, z)
    offsets = NormalSamplingSpec.from_offsets([-2.0, 0.0, 2.0]).offsets_um

    report = validate_surface_geometry(
        surface,
        volume_shape_zyx=(12, 15, 16),
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        normal_offsets_um=offsets,
    )

    assert report.valid_vertex_count == 30
    assert report.in_bounds_vertex_fraction == 1.0
    assert report.normal_valid_vertex_fraction == 1.0
    assert report.median_neighbor_distance_um == 1.0
    assert report.median_quad_area_um2 == 1.0
    assert report.metric_distortion_p95 == 1.0
    assert report.degenerate_quad_count == 0
    assert report.folded_quad_count == 0
    assert report.abrupt_normal_flip_edge_count == 0
    assert report.neighbor_wrap_risk_edge_count == 0
    assert report.offset_sample_in_bounds_fraction == 1.0
    assert report.warnings == ()


def test_geometry_validator_flags_bounds_jumps_distortion_and_normal_flips() -> None:
    rows, _ = np.mgrid[:5, :5]
    # Direction reverses abruptly after column 2.  The long 2 -> -2 edge is a
    # neighbor-wrap risk; local x direction and therefore +z normal also flips.
    x_line = np.asarray([0.0, 1.0, 2.0, -2.0, -3.0])
    x = np.broadcast_to(x_line, (5, 5)).copy()
    y = rows.astype(np.float64) + 2.0
    z = np.full((5, 5), 4.0)
    surface = SurfaceGrid.from_tifxyz(x, y, z)

    report = validate_surface_geometry(
        surface,
        volume_shape_zyx=(10, 10, 10),
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        continuity_factor=2.0,
        neighbor_wrap_factor=3.0,
        metric_distortion_limit=2.0,
        abrupt_normal_angle_degrees=80.0,
    )

    assert report.in_bounds_vertex_fraction < 1.0
    assert report.discontinuity_edge_count > 0
    assert report.neighbor_wrap_risk_edge_count > 0
    assert report.distorted_quad_count > 0
    assert report.abrupt_normal_flip_edge_count > 0
    assert any("outside the volume" in warning for warning in report.warnings)
    assert any("neighbor-wrap-risk" in warning for warning in report.warnings)
    assert any("normal flips" in warning for warning in report.warnings)


def test_geometry_validator_counts_degenerate_quads_and_offset_bounds() -> None:
    x, y, z = make_xyz_plane(4, 4, z=1.0, x_origin=1.0, y_origin=1.0)
    # Collapse one full grid cell to zero width.
    x[:, 2] = x[:, 1]
    surface = SurfaceGrid.from_tifxyz(x, y, z)

    report = validate_surface_geometry(
        surface,
        volume_shape_zyx=(5, 8, 8),
        voxel_size_zyx_um=(1.0, 1.0, 1.0),
        normal_offsets_um=(-2.0, 0.0, 2.0),
    )

    assert report.degenerate_quad_count > 0
    assert report.offset_sample_in_bounds_fraction < 1.0
    assert any("degenerate quads" in warning for warning in report.warnings)
    assert any("normal-offset samples" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    "bad_coordinates",
    [np.zeros((2, 2)), np.zeros((2, 3, 4))],
)
def test_trilinear_rejects_coordinate_arrays_without_xyz_axis(
    bad_coordinates: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="final .*x, y, z.* axis"):
        sample_trilinear_zyx(np.zeros((3, 3, 3)), bad_coordinates)
