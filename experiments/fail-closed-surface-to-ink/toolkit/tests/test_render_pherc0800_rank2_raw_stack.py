from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from native_surface_sampler import SurfaceGrid  # noqa: E402
from render_pherc0800_rank2_raw_stack import (  # noqa: E402
    CachedPublicZarrArray,
    model_window_mappings,
    physical_area_cm2,
    quad_union_vertex_mask,
    scalable_topology_report,
)
from sample_raw_ct_seed_cube import ZarrV2ArraySpec  # noqa: E402


def test_scalable_topology_rectangle_and_annulus() -> None:
    rectangle = scalable_topology_report(np.ones((5, 7), dtype=bool))
    assert not rectangle["nonmanifold_risk"]
    assert rectangle["euler_characteristic"] == 1
    assert rectangle["boundary_loops_if_manifold"] == 1
    assert rectangle["holes_if_manifold"] == 0

    annulus = np.ones((9, 9), dtype=bool)
    annulus[3:6, 3:6] = False
    report = scalable_topology_report(annulus)
    assert not report["nonmanifold_risk"]
    assert report["euler_characteristic"] == 0
    assert report["boundary_loops_if_manifold"] == 2
    assert report["holes_if_manifold"] == 1


def test_scalable_topology_flags_bowtie_and_orphan() -> None:
    quads = np.zeros((4, 4), dtype=bool)
    quads[0:2, 0:2] = True
    quads[2:4, 2:4] = True
    bowtie = scalable_topology_report(quad_union_vertex_mask(quads))
    assert bowtie["nonmanifold_risk"]
    assert bowtie["bowtie_vertex_count"] == 1

    mask = np.zeros((7, 7), dtype=bool)
    mask[:3, :3] = True
    mask[6, 6] = True
    orphan = scalable_topology_report(mask)
    assert orphan["nonmanifold_risk"]


def test_physical_area_on_unit_voxel_plane() -> None:
    rows, columns = np.mgrid[:4, :6]
    surface = SurfaceGrid.from_tifxyz(
        columns,
        rows,
        np.ones((4, 6)),
        mask=np.ones((4, 6), dtype=bool),
    )
    expected = 3 * 5 * 8.64 * 8.64 / 100_000_000.0
    assert np.isclose(physical_area_cm2(surface), expected)


def test_exact_93_and_62_layer_sign_order_mappings() -> None:
    mappings = model_window_mappings()
    full = mappings["full_93_layer_sign_order_equivalences"]
    assert full["positive_direct"] == full["negative_reversed"]
    assert full["positive_reversed"] == full["negative_direct"]
    windows = mappings["latest_62_layer_windows"]
    assert all(len(indices) == 62 for indices in windows.values())
    assert windows["negative_low_direct"] == list(
        reversed(windows["positive_high_direct"])
    )
    assert windows["negative_high_direct"] == list(
        reversed(windows["positive_low_direct"])
    )


def test_cached_public_zarr_array_fetches_each_payload_once(tmp_path: Path) -> None:
    spec = ZarrV2ArraySpec(
        shape_zyx=(4, 4, 4),
        chunks_zyx=(2, 2, 2),
        dtype_string=np.dtype(np.uint8).str,
        order="C",
        fill_value=0,
        dimension_separator="/",
        compressor=None,
        filters=None,
    )
    calls: list[str] = []
    array = CachedPublicZarrArray(
        root_url="https://example.invalid/raw.zarr",
        array_path="0",
        spec=spec,
        cache_directory=tmp_path / "cache",
        blosc_path=tmp_path / "unused-blosc",
        decoded_lru_chunks=1,
    )

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        index = tuple(int(value) for value in url.split("/")[-3:])
        value = 1 + index[0] * 4 + index[1] * 2 + index[2]
        return np.full((2, 2, 2), value, dtype=np.uint8).tobytes()

    array._fetch_public = fake_fetch  # type: ignore[method-assign]
    first = array[0:3, 0:3, 0:3]
    second = array[0:3, 0:3, 0:3]
    np.testing.assert_array_equal(first, second)
    assert len(calls) == 8
    assert array.network_fetch_count == 8
    assert array.manifest()["unique_chunk_count"] == 8
