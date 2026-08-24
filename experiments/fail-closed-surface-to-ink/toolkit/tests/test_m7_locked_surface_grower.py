from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import m7_locked_surface_grower as grower  # noqa: E402
from m7_locked_surface_grower import (  # noqa: E402
    DEFAULT_M7_ROOT,
    DEFAULT_VOLUME_SHAPE_ZYX,
    GrowConfig,
    build_parser,
    float_transect,
    grow_complete_rings,
    load_public_sampler,
    project_target_to_m7,
    projections_are_distinct_sheets,
    tangent_basis_zyx,
    triangulated_area_cm2,
)


class ArrayAccessor:
    def __init__(self, foreground: np.ndarray) -> None:
        self.foreground = np.asarray(foreground, dtype=bool)
        self.shape_zyx = self.foreground.shape
        self.read_count = 0

    def read_roi(self, lower_zyx, upper_zyx):
        lower = np.asarray(lower_zyx, dtype=int)
        upper = np.asarray(upper_zyx, dtype=int)
        self.read_count += 1
        return self.foreground[
            tuple(slice(int(left), int(right)) for left, right in zip(lower, upper))
        ].copy()


def plane_volume(
    *,
    shape=(161, 161, 161),
    z0=79,
    z1=82,
    y0=30,
    y1=131,
    x0=30,
    x1=131,
) -> np.ndarray:
    volume = np.zeros(shape, dtype=bool)
    volume[z0:z1, y0:y1, x0:x1] = True
    return volume


def small_config(**overrides) -> GrowConfig:
    values = dict(
        target_radius=2,
        minimum_output_radius=2,
        spacing_voxels=10.0,
        proposal_snap_radius=3.0,
        pca_radius=3,
        pca_min_points=10,
        transect_half_length=8,
        maximum_initial_run_center_offset=1.0,
        maximum_projected_run_center_offset=1.0,
        maximum_projection_distance=4.0,
        maximum_proposal_spread=3.0,
        maximum_candidates_evaluated=48,
    )
    values.update(overrides)
    return GrowConfig(**values)


def test_tangent_basis_is_orthonormal_and_tangent() -> None:
    normal = np.array([0.3, -0.4, 0.866025403784])
    normal /= np.linalg.norm(normal)
    row, column = tangent_basis_zyx(normal)
    np.testing.assert_allclose(np.linalg.norm(row), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(column), 1.0, atol=1e-12)
    np.testing.assert_allclose(row @ column, 0.0, atol=1e-12)
    np.testing.assert_allclose(row @ normal, 0.0, atol=1e-12)
    np.testing.assert_allclose(column @ normal, 0.0, atol=1e-12)


def test_float_transect_requires_one_centered_run() -> None:
    foreground = np.zeros((31, 11, 11), dtype=bool)
    foreground[14:17, :, :] = True
    clean = float_transect(foreground, (15, 5, 5), (1, 0, 0), 10)
    assert clean["pass"]
    assert clean["runs"] == [[-1, 1]]
    foreground[22:24, :, :] = True
    ambiguous = float_transect(foreground, (15, 5, 5), (1, 0, 0), 10)
    assert not ambiguous["pass"]
    assert ambiguous["reason"] == "competing_foreground_run"


def test_projection_rejects_parallel_sheet_ambiguity() -> None:
    volume = plane_volume(z0=79, z1=82)
    volume[86:89, 30:131, 30:131] = True
    accessor = ArrayAccessor(volume)
    result = project_target_to_m7(
        accessor,
        (80, 80, 80),
        (1, 0, 0),
        small_config(transect_half_length=12),
        inward_neighbor_count=1,
    )
    assert not result.passed
    assert "unique_centered" in result.reason


def test_ambiguity_uses_normal_not_tangent_separation() -> None:
    normal = (1.0, 0.0, 0.0)
    assert projections_are_distinct_sheets((0, 0, 0), (3, 0, 0), normal, 2.5)
    assert not projections_are_distinct_sheets((0, 0, 0), (0, 4, 4), normal, 2.5)


def test_complete_ring_growth_on_planar_m7_is_deterministic() -> None:
    accessor = ArrayAccessor(plane_volume())
    config = small_config()
    first = grow_complete_rings(accessor, (80, 80, 80), config, voxel_um=1.0)
    second = grow_complete_rings(ArrayAccessor(plane_volume()), (80, 80, 80), config, voxel_um=1.0)
    assert first.passed_minimum_size
    assert first.accepted_radius == 2
    assert first.points_zyx.shape == (5, 5, 3)
    np.testing.assert_allclose(first.points_zyx, second.points_zyx, atol=0, rtol=0)
    assert all(record["pass"] for record in first.ring_records)
    for corner in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        assert first.nodes[corner].inward_neighbor_count == 3
    assert triangulated_area_cm2(first.points_zyx, 1.0) > 0


def test_complete_ring_growth_accepts_a_projection_strategy() -> None:
    calls = 0

    def counted_projector(*args, **kwargs):
        nonlocal calls
        calls += 1
        return project_target_to_m7(*args, **kwargs)

    result = grow_complete_rings(
        ArrayAccessor(plane_volume()),
        (80, 80, 80),
        small_config(),
        voxel_um=1.0,
        projector=counted_projector,
    )
    assert result.accepted_radius == 2
    assert calls == 25


def test_failed_outer_ring_is_not_partially_materialized() -> None:
    volume = plane_volume(y0=65, y1=96, x0=65, x1=96)
    result = grow_complete_rings(
        ArrayAccessor(volume),
        (80, 80, 80),
        small_config(),
        voxel_um=1.0,
    )
    assert result.accepted_radius == 1
    assert result.points_zyx.shape == (3, 3, 3)
    assert not result.passed_minimum_size
    assert result.ring_records[0]["pass"]
    assert not result.ring_records[1]["pass"]
    assert "ring_2_rejected" in result.stop_reason


def test_parser_preserves_defaults_and_accepts_target_volume_override() -> None:
    defaults = build_parser().parse_args([])
    assert tuple(defaults.volume_shape_zyx) == DEFAULT_VOLUME_SHAPE_ZYX
    assert defaults.target_volume == Path(DEFAULT_M7_ROOT).name

    overridden = build_parser().parse_args(
        [
            "--volume-shape-zyx",
            "18977",
            "6844",
            "6844",
            "--target-volume",
            "PHerc1203-eligible-9.362um",
            "--voxel-um",
            "9.362",
        ]
    )
    assert tuple(overridden.volume_shape_zyx) == (18977, 6844, 6844)
    assert overridden.target_volume == "PHerc1203-eligible-9.362um"
    assert overridden.voxel_um == 9.362


def test_public_sampler_shape_gate_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    array_metadata = {
        "zarr_format": 2,
        "shape": [18977, 6844, 6844],
        "chunks": [192, 192, 192],
        "dtype": "|u1",
        "order": "C",
        "fill_value": 0,
        "dimension_separator": "/",
        "compressor": {"id": "blosc", "cname": "zstd"},
        "filters": None,
    }

    def fake_fetch_json(url: str):
        if url.endswith("/.zgroup"):
            document = {"zarr_format": 2}
        elif url.endswith("/0/.zarray"):
            document = array_metadata
        elif url.endswith("/.zattrs"):
            document = {}
        else:  # pragma: no cover - guards unexpected metadata requests
            raise AssertionError(url)
        return document, {"url": url, "bytes": 1, "sha256": "0" * 64}

    monkeypatch.setattr(grower, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(grower, "validate_metadata_axes", lambda *_: None)
    codec = tmp_path / "libblosc.test"
    codec.write_bytes(b"test codec identity")

    sampler, provenance = load_public_sampler(
        "https://example.invalid/pherc1203-m7",
        "0",
        codec,
        (18977, 6844, 6844),
    )
    assert sampler.shape_zyx == (18977, 6844, 6844)
    assert provenance["array_spec"]["shape_zyx"] == (18977, 6844, 6844)

    with pytest.raises(ValueError, match="unexpected public m7 shape"):
        load_public_sampler(
            "https://example.invalid/pherc1203-m7", "0", codec
        )
