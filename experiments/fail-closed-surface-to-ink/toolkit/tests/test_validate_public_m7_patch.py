from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import sample_trilinear_zyx  # noqa: E402
from sample_raw_ct_seed_cube import ZarrV2ArraySpec  # noqa: E402
from validate_public_m7_patch import (  # noqa: E402
    PublicZarrChunkSampler,
    closest_point_on_triangle,
    decide_intermediate_gate,
    evaluate_valid_vertex_transects,
    generation_support_summary,
    seed_surface_alignment,
    support_summary,
)
from native_surface_sampler import SurfaceGrid  # noqa: E402


def uncompressed_spec() -> ZarrV2ArraySpec:
    return ZarrV2ArraySpec.from_metadata(
        {
            "zarr_format": 2,
            "shape": [8, 8, 8],
            "chunks": [4, 4, 4],
            "dtype": "|u1",
            "order": "C",
            "fill_value": 0,
            "filters": None,
            "compressor": None,
            "dimension_separator": "/",
        }
    )


def test_public_sampler_matches_native_and_fetches_each_needed_chunk_once() -> None:
    volume = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8) % 256
    volume = volume.astype(np.uint8)
    requests: list[str] = []

    def fetcher(url: str) -> bytes:
        requests.append(url)
        zc, yc, xc = (int(value) for value in url.rsplit("/", 3)[-3:])
        return volume[
            zc * 4 : (zc + 1) * 4,
            yc * 4 : (yc + 1) * 4,
            xc * 4 : (xc + 1) * 4,
        ].tobytes()

    sampler = PublicZarrChunkSampler(
        root_url="https://example.test/data.zarr",
        array_path="0",
        spec=uncompressed_spec(),
        fetcher=fetcher,
    )
    points = np.array([[3.7, 3.8, 3.9], [4.2, 4.3, 4.4], [3.9, 4.1, 2.5]])
    valid = np.ones(3, dtype=bool)
    actual, actual_valid = sampler.sample(points, point_valid=valid)
    expected = sample_trilinear_zyx(volume, points, point_valid=valid)
    np.testing.assert_allclose(actual, expected.values, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(actual_valid, expected.valid)
    sampler.sample(points, point_valid=valid)
    assert len(requests) == len(set(requests))
    assert sampler.manifest()["each_chunk_fetched_at_most_once"]


def test_deterministic_transects_choose_only_valid_vertices() -> None:
    frames = np.zeros((31, 8, 8), dtype=np.uint8)
    frames[14:17] = 255
    valid = np.ones((8, 8), dtype=bool)
    valid[0, 0] = False
    frames[2, 0, 0] = 255
    first = evaluate_valid_vertex_transects(frames, valid, seed=1447, sample_count=50)
    second = evaluate_valid_vertex_transects(frames, valid, seed=1447, sample_count=50)
    assert first["pass"]
    assert first["chosen_flat_indices"] == second["chosen_flat_indices"]
    assert 0 not in first["chosen_flat_indices"]


def test_support_summary_uses_any_plus_minus_two_and_reports_offset_zero() -> None:
    frames = np.zeros((5, 2, 2), dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    frames[0, 0, 0] = 1
    frames[2, 0, 1] = 1
    frames[4, 1, 0] = 1
    result = support_summary(frames, (-2, -1, 0, 1, 2), valid)
    assert result["supported_vertex_count"] == 3
    assert result["support_fraction"] == 0.75
    assert result["offset0_supported_vertex_count"] == 1
    assert result["unsupported_row_column"] == [[1, 1]]


def test_generation_summary_treats_minimum_valid_generation_as_seed_core() -> None:
    generations = np.array([[0, 1], [2, 3]], dtype=np.uint16)
    valid = np.array([[False, True], [True, True]])
    support = np.full((5, 2, 2), 255, dtype=np.uint8)
    transects = np.full((31, 2, 2), 255, dtype=np.uint8)
    result = generation_support_summary(
        generations, valid, support, (-2, -1, 0, 1, 2), transects
    )
    assert result["generation_zero_valid_vertex_count"] == 0
    assert result["minimum_valid_generation_interpreted_as_seed_core"] == 1
    assert result["seed_core"]["vertex_count"] == 1
    assert result["seed_core"]["support"]["support_fraction"] == 1.0


def test_intermediate_gate_is_fail_closed() -> None:
    go, gates = decide_intermediate_gate(
        hard_geometry_pass=True,
        sampling_complete=True,
        support_fraction=0.95,
        minimum_support=0.90,
        seed_support_fraction=1.0,
        minimum_seed_support=1.0,
        transects_pass=True,
    )
    assert go
    assert all(gate["pass"] for gate in gates)
    no_go, failed = decide_intermediate_gate(
        hard_geometry_pass=True,
        sampling_complete=True,
        support_fraction=0.89,
        minimum_support=0.90,
        seed_support_fraction=1.0,
        minimum_seed_support=1.0,
        transects_pass=True,
    )
    assert not no_go
    assert next(gate for gate in failed if gate["name"] == "five_offset_m7_support")["pass"] is False


def test_seed_alignment_measures_distance_to_valid_triangle_interior() -> None:
    surface = SurfaceGrid.from_tifxyz(
        [[0, 1], [0, 1]],
        [[0, 0], [1, 1]],
        [[0, 0], [0, 0]],
        mask=np.ones((2, 2), dtype=bool),
    )
    result = seed_surface_alignment(surface, (0.25, 0.75, 3.0), voxel_um=2.0)
    assert result["evaluated"]
    assert result["distance_voxels"] == 3.0
    assert result["distance_um"] == 6.0
    np.testing.assert_allclose(result["nearest_surface_point_xyz"], (0.25, 0.75, 0.0))


def test_closest_point_on_triangle_uses_nearest_vertex_outside() -> None:
    actual = closest_point_on_triangle(
        (3, 3, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)
    )
    np.testing.assert_allclose(actual, (0.5, 0.5, 0.0))
