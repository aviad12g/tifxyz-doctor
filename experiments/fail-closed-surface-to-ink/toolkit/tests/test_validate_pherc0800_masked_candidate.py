from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from validate_pherc0800_masked_candidate import (  # noqa: E402
    active_quads,
    conservative_manifold_subset,
    materialize_masked_tifxyz,
    quad_union_vertex_mask,
    topology_report,
)


def test_topology_rectangle_is_one_disk() -> None:
    mask = np.ones((5, 7), dtype=bool)
    report = topology_report(mask)
    assert report["connected_quad_component_count"] == 1
    assert report["euler_characteristic"] == 1
    assert report["boundary_loops_if_manifold"] == 1
    assert report["holes_if_manifold"] == 0
    assert not report["nonmanifold_risk"]


def test_topology_annulus_has_one_hole_and_two_loops() -> None:
    quads = np.ones((5, 5), dtype=bool)
    quads[2, 2] = False
    mask = quad_union_vertex_mask(quads)
    # A single missing quad is not representable by a pure vertex mask because
    # its four vertices are needed by the ring; the encoding proof detects fill.
    assert active_quads(mask)[2, 2]

    vertex_mask = np.ones((7, 7), dtype=bool)
    vertex_mask[2:5, 2:5] = False
    report = topology_report(vertex_mask)
    assert report["connected_quad_component_count"] == 1
    assert report["euler_characteristic"] == 0
    assert report["boundary_loops_if_manifold"] == 2
    assert report["holes_if_manifold"] == 1
    assert not report["nonmanifold_risk"]


def test_topology_flags_diagonal_vertex_pinch() -> None:
    quads = np.zeros((2, 2), dtype=bool)
    quads[0, 0] = True
    quads[1, 1] = True
    mask = quad_union_vertex_mask(quads)
    report = topology_report(mask)
    assert report["nonmanifold_risk"]
    assert report["boundary_degree_histogram"].get("4") == 1
    assert report["bowtie_vertex_count"] == 1
    assert report["bowtie_vertices"][0]["row_column"] == [1, 1]


def test_conservative_repair_deletes_pinch_and_is_idempotent() -> None:
    quads = np.zeros((5, 5), dtype=bool)
    quads[0:2, 0:2] = True
    quads[2:5, 2:5] = True
    mask = quad_union_vertex_mask(quads)
    area = np.ones(quads.shape, dtype=np.float64)
    repaired, report = conservative_manifold_subset(mask, area)
    assert report["iteration_count"] == 1
    assert report["strict_quad_subset_of_input"]
    assert report["idempotent"]
    assert report["iterations"][0][
        "simultaneously_deleted_bowtie_vertices_row_column"
    ] == [[2, 2]]
    assert not topology_report(repaired)["nonmanifold_risk"]
    assert not np.any(active_quads(repaired) & ~active_quads(mask))
    repeated, repeated_report = conservative_manifold_subset(repaired, area)
    np.testing.assert_array_equal(repeated, repaired)
    assert repeated_report["iteration_count"] == 0


def test_quad_union_roundtrip_for_encodable_component() -> None:
    quads = np.zeros((4, 5), dtype=bool)
    quads[1:4, 1:3] = True
    mask = quad_union_vertex_mask(quads)
    np.testing.assert_array_equal(active_quads(mask), quads)


def test_materialization_copies_coordinates_and_adds_only_mask(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows, columns = np.mgrid[:5, :6]
    arrays = {
        "x": (columns + 100).astype(np.float32),
        "y": (rows + 200).astype(np.float32),
        "z": np.full((5, 6), 300, dtype=np.float32),
    }
    for name, array in arrays.items():
        tifffile.imwrite(source / f"{name}.tif", array)
    (source / "meta.json").write_text(
        json.dumps({"uuid": "source", "scale": [1, 1], "seed": [0, 0, 0]}),
        encoding="utf-8",
    )
    mask = np.zeros((5, 6), dtype=bool)
    mask[1:5, 1:4] = True
    quad_area = np.ones((4, 5), dtype=float) * 0.01
    source_bytes = {
        name: (source / name).read_bytes() for name in ("x.tif", "y.tif", "z.tif")
    }
    report = materialize_masked_tifxyz(
        source,
        tmp_path / "destination",
        mask,
        quad_area_cm2=quad_area,
        role="test",
        lineage={"fixed": True},
        voxel_um=10.0,
    )
    assert report["no_bridging_proof"]
    assert report["active_quad_count"] == 6
    assert np.isclose(report["physical_area_cm2"], 0.06)
    metadata = json.loads((tmp_path / "destination/meta.json").read_text())
    assert np.isclose(metadata["area_vx2"], 60_000.0)
    for name, payload in source_bytes.items():
        assert (tmp_path / "destination" / name).read_bytes() == payload
    materialized_mask = tifffile.imread(tmp_path / "destination/mask.tif") != 0
    np.testing.assert_array_equal(materialized_mask, mask)
