#!/usr/bin/env python3
"""Materialize and freshly validate a fixed PHerc0800 candidate as masked TIFFXYZ.

The preflight-selected connected quad component is fixed before this program
runs.  Coordinates are copied byte-for-byte; only an explicit vertex mask and
lineage metadata are added.  The program proves that the vertex mask encodes
exactly the selected quads (no implicit bridging), recomputes mask-aware
boundary/hole normals, resamples official public m7 over -15..+15, and measures
the largest freshly centered connected-quad component.  It never moves,
clamps, fills, or interpolates a source vertex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from audit_pherc0800_published_segments import (
    DiskPayloadFetcher,
    connected_quad_components_summary,
    mark_distorted_vertices,
    quad_component_vertex_mask,
    run_metrics_summary,
    selected_region_diagnostics,
    selected_run_metrics,
)
from numpy.typing import NDArray
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    dilate_one_cell,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import ZarrV2ArraySpec, fetch_json, sha256_file
from sweep_public_m7_uniform_offsets import hard_geometry_summary
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    _array_sha256,
    _atomic_json,
    support_summary,
    validate_metadata_axes,
)

LONG_ID = "20251028220042-auto_grown_20251028220042762"
DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/first-letters-geometry/pherc0800-published-segments-preflight"
)
DEFAULT_PREFLIGHT_REPORT = DEFAULT_PREFLIGHT_DIR / (
    "pherc0800-published-segments-preflight.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/"
    "pherc0800-rank2-masked-validation-v2-topology-repaired"
)
DEFAULT_MINIMUM_AREA_CM2 = 0.5
VOXEL_UM = 8.64


Vertex = tuple[int, int]
Edge = tuple[Vertex, Vertex]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def active_quads(vertex_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    mask = np.asarray(vertex_mask, dtype=bool)
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def quad_union_vertex_mask(quad_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    selected = np.asarray(quad_mask, dtype=bool)
    result = np.zeros((selected.shape[0] + 1, selected.shape[1] + 1), dtype=bool)
    result[:-1, :-1] |= selected
    result[:-1, 1:] |= selected
    result[1:, :-1] |= selected
    result[1:, 1:] |= selected
    return result


def canonical_edge(first: Vertex, second: Vertex) -> Edge:
    return (first, second) if first < second else (second, first)


def topology_report(vertex_mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Topology of the implicit all-four-valid quad complex."""

    mask = np.asarray(vertex_mask, dtype=bool)
    quads = active_quads(mask)
    edge_incidence: Counter[Edge] = Counter()
    used_vertices: set[Vertex] = set()
    for row, column in np.argwhere(quads):
        row, column = int(row), int(column)
        vertices = (
            (row, column),
            (row, column + 1),
            (row + 1, column + 1),
            (row + 1, column),
        )
        used_vertices.update(vertices)
        for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=True):
            edge_incidence[canonical_edge(first, second)] += 1
    boundary_edges = [edge for edge, count in edge_incidence.items() if count == 1]
    boundary_adjacency: dict[Vertex, set[Vertex]] = {}
    for first, second in boundary_edges:
        boundary_adjacency.setdefault(first, set()).add(second)
        boundary_adjacency.setdefault(second, set()).add(first)
    boundary_degrees = Counter(
        len(neighbors) for neighbors in boundary_adjacency.values()
    )
    anomalous_boundary_vertices = sorted(
        vertex
        for vertex, neighbors in boundary_adjacency.items()
        if len(neighbors) != 2
    )
    bowtie_vertices: list[dict[str, Any]] = []
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            nw = row > 0 and column > 0 and quads[row - 1, column - 1]
            ne = row > 0 and column < quads.shape[1] and quads[row - 1, column]
            sw = row < quads.shape[0] and column > 0 and quads[row, column - 1]
            se = (
                row < quads.shape[0]
                and column < quads.shape[1]
                and quads[row, column]
            )
            pattern = (bool(nw), bool(ne), bool(sw), bool(se))
            if pattern in (
                (True, False, False, True),
                (False, True, True, False),
            ):
                bowtie_vertices.append(
                    {
                        "row_column": [row, column],
                        "incident_quadrants_nw_ne_sw_se": [
                            int(value) for value in pattern
                        ],
                    }
                )
    unseen = set(boundary_adjacency)
    boundary_components = 0
    while unseen:
        boundary_components += 1
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in boundary_adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    quad_components, _labels = connected_quad_components_summary(
        mask, np.ones(quads.shape, dtype=np.float64)
    )
    vertices = len(used_vertices)
    edges = len(edge_incidence)
    faces = int(np.count_nonzero(quads))
    euler = vertices - edges + faces
    component_count = int(quad_components["component_count"])
    manifold_boundary = bool(
        boundary_edges
        and set(boundary_degrees) == {2}
        and max(edge_incidence.values(), default=0) <= 2
    )
    holes = component_count - euler if manifold_boundary else None
    expected_boundary_loops = (
        component_count + int(holes) if holes is not None else None
    )
    reasons: list[str] = []
    if component_count != 1:
        reasons.append("quad complex is not one connected component")
    if any(count > 2 for count in edge_incidence.values()):
        reasons.append("one or more edges have incidence greater than two")
    if set(boundary_degrees) != {2}:
        reasons.append("boundary graph contains open ends or branch/pinch vertices")
    if holes is not None and holes < 0:
        reasons.append("Euler-derived hole count is negative")
    if (
        expected_boundary_loops is not None
        and boundary_components != expected_boundary_loops
    ):
        reasons.append("boundary-loop count disagrees with Euler characteristic")
    return {
        "complex": "axis-aligned stored-grid quads with all four vertices valid",
        "vertex_count_used_by_quads": vertices,
        "edge_count": edges,
        "quad_face_count": faces,
        "euler_characteristic": euler,
        "connected_quad_component_count": component_count,
        "boundary_edge_count": len(boundary_edges),
        "boundary_graph_component_count": boundary_components,
        "boundary_degree_histogram": {
            str(degree): int(count)
            for degree, count in sorted(boundary_degrees.items())
        },
        "anomalous_boundary_vertices_row_column": [
            list(vertex) for vertex in anomalous_boundary_vertices
        ],
        "bowtie_vertex_count": len(bowtie_vertices),
        "bowtie_vertices": bowtie_vertices,
        "boundary_loops_if_manifold": boundary_components
        if manifold_boundary
        else None,
        "holes_if_manifold": holes,
        "maximum_edge_incidence": max(edge_incidence.values(), default=0),
        "nonmanifold_risk": bool(reasons),
        "nonmanifold_risk_reasons": reasons,
        "self_intersection_status": "unknown_not_computed",
    }


def conservative_manifold_subset(
    vertex_mask: NDArray[np.bool_],
    quad_area_cm2: NDArray[np.float64],
    *,
    maximum_iterations: int = 100,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    """Delete every diagonal pinch and retain only the largest resulting sheet.

    The transform is deliberately subtractive: it never adds a vertex or quad,
    and component selection is deterministic (physical area, quad count, then
    the row-major component id used by the connected-component implementation).
    """

    original = np.asarray(vertex_mask, dtype=bool)
    if quad_area_cm2.shape != (original.shape[0] - 1, original.shape[1] - 1):
        raise ValueError("quad area shape differs from vertex mask")
    original_quads = active_quads(original)
    current = original.copy()
    iterations: list[dict[str, Any]] = []
    for iteration in range(maximum_iterations + 1):
        topology = topology_report(current)
        if not topology["nonmanifold_risk"]:
            repaired_quads = active_quads(current)
            if np.any(repaired_quads & ~original_quads):
                raise RuntimeError("topology repair added a source quad")
            if not np.array_equal(active_quads(quad_union_vertex_mask(repaired_quads)), repaired_quads):
                raise RuntimeError("repaired quad set is not exactly vertex-mask encodable")
            return current, {
                "algorithm": (
                    "simultaneously delete every diagonal 1001/0110 pinch vertex; "
                    "recompute induced quads; retain the component ranked by physical "
                    "area descending, quad count descending, then row-major component; "
                    "rebuild vertices from quad union; repeat to fixed point"
                ),
                "coordinate_policy": "strictly subtractive; no add/fill/move/clamp",
                "iteration_count": len(iterations),
                "iterations": iterations,
                "initial_vertex_count": int(np.count_nonzero(original)),
                "initial_quad_count": int(np.count_nonzero(original_quads)),
                "initial_area_cm2": physical_area_from_mask(original, quad_area_cm2),
                "repaired_vertex_count": int(np.count_nonzero(current)),
                "repaired_quad_count": int(np.count_nonzero(repaired_quads)),
                "repaired_area_cm2": physical_area_from_mask(current, quad_area_cm2),
                "removed_vertex_count": int(np.count_nonzero(original & ~current)),
                "removed_quad_count": int(
                    np.count_nonzero(original_quads & ~repaired_quads)
                ),
                "removed_area_cm2": float(
                    physical_area_from_mask(original, quad_area_cm2)
                    - physical_area_from_mask(current, quad_area_cm2)
                ),
                "strict_quad_subset_of_input": bool(
                    np.all(~repaired_quads | original_quads)
                ),
                "idempotent": True,
                "fixed_point_topology": topology,
            }
        if iteration >= maximum_iterations:
            raise RuntimeError("topology repair did not reach a manifold fixed point")
        bowties = topology["bowtie_vertices"]
        if not bowties:
            raise RuntimeError(
                "topology is nonmanifold but has no repairable diagonal pinch"
            )
        before_quads = active_quads(current)
        deleted = current.copy()
        coordinates = [tuple(item["row_column"]) for item in bowties]
        for row, column in coordinates:
            deleted[int(row), int(column)] = False
        after_deletion_quads = active_quads(deleted)
        if np.any(after_deletion_quads & ~before_quads):
            raise RuntimeError("pinch deletion unexpectedly added a quad")
        components, labels = connected_quad_components_summary(
            deleted, quad_area_cm2
        )
        largest = components["largest_component"]
        if largest is None:
            raise RuntimeError("pinch deletion removed every positive-area component")
        retained_quads = labels == int(largest["component_id"])
        next_mask = quad_union_vertex_mask(retained_quads)
        if not np.array_equal(active_quads(next_mask), retained_quads):
            raise RuntimeError(
                "retained component cannot be encoded without implicit quad additions"
            )
        if np.any(retained_quads & ~original_quads):
            raise RuntimeError("component cleanup added a source quad")
        iterations.append(
            {
                "iteration": iteration + 1,
                "simultaneously_deleted_bowtie_vertices_row_column": [
                    list(value) for value in coordinates
                ],
                "quad_count_before": int(np.count_nonzero(before_quads)),
                "quad_count_after_deletion": int(
                    np.count_nonzero(after_deletion_quads)
                ),
                "component_count_after_deletion": int(
                    components["component_count"]
                ),
                "retained_component": largest,
                "retained_quad_count": int(np.count_nonzero(retained_quads)),
                "retained_area_cm2": float(largest["area_cm2"]),
                "dropped_or_orphaned_vertex_count": int(
                    np.count_nonzero(deleted & ~next_mask)
                ),
            }
        )
        current = next_mask
    raise AssertionError("unreachable")


def physical_area_from_mask(
    vertex_mask: NDArray[np.bool_], quad_area_cm2: NDArray[np.float64]
) -> float:
    return float(np.sum(np.asarray(quad_area_cm2)[active_quads(vertex_mask)]))


def materialize_masked_tifxyz(
    source_dir: Path,
    destination: Path,
    vertex_mask: NDArray[np.bool_],
    *,
    quad_area_cm2: NDArray[np.float64],
    role: str,
    lineage: Mapping[str, Any],
    voxel_um: float = VOXEL_UM,
    default_uuid: str = LONG_ID,
) -> dict[str, Any]:
    """Copy coordinates byte-exactly and add an explicit mask and lineage metadata."""

    source_asset = load_tifxyz_asset(source_dir, resolution="stored")
    mask = np.asarray(vertex_mask, dtype=bool)
    if mask.shape != source_asset.surface.shape:
        raise ValueError("materialization mask shape differs from source TIFFXYZ")
    if np.any(mask & ~source_asset.surface.valid):
        raise ValueError("materialization mask selects an invalid source vertex")
    selected_quads = active_quads(mask)
    if not np.array_equal(
        active_quads(quad_union_vertex_mask(selected_quads)), selected_quads
    ):
        raise ValueError("vertex mask would implicitly bridge or fill unselected quads")
    destination.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for filename in ("x.tif", "y.tif", "z.tif"):
        source = source_dir / filename
        target = destination / filename
        shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"coordinate copy hash mismatch for {filename}")
        files.append(
            {
                "filename": filename,
                "source_path": str(source.resolve()),
                "path": str(target.resolve()),
                "sha256": sha256_file(target),
                "byte_exact_source_copy": True,
            }
        )
    mask_path = destination / "mask.tif"
    tifffile.imwrite(mask_path, mask.astype(np.uint8) * 255, photometric="minisblack")
    points = source_asset.surface.points_xyz[mask]
    metadata = dict(source_asset.metadata)
    source_uuid = str(metadata.get("uuid", default_uuid))
    metadata.update(
        {
            "uuid": f"{source_uuid}_{role}",
            "area_cm2": physical_area_from_mask(mask, quad_area_cm2),
            "area_vx2": float(
                physical_area_from_mask(mask, quad_area_cm2)
                * 1.0e8
                / (float(voxel_um) * float(voxel_um))
            ),
            "bbox": [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()],
            "source": "fixed_mask_only_no_coordinate_modification",
            "masked_component_lineage": dict(lineage),
        }
    )
    meta_path = destination / "meta.json"
    _safe_write_bytes(
        meta_path,
        (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )
    files.extend(
        [
            {
                "filename": "mask.tif",
                "path": str(mask_path.resolve()),
                "sha256": sha256_file(mask_path),
                "array_content_sha256": _array_sha256(mask),
            },
            {
                "filename": "meta.json",
                "path": str(meta_path.resolve()),
                "sha256": sha256_file(meta_path),
            },
        ]
    )
    reloaded = load_tifxyz_asset(destination, resolution="stored")
    if not np.array_equal(reloaded.surface.valid, mask):
        raise RuntimeError("materialized TIFFXYZ mask changed on reload")
    for original, copied in zip(
        (source_asset.surface.x, source_asset.surface.y, source_asset.surface.z),
        (reloaded.surface.x, reloaded.surface.y, reloaded.surface.z),
        strict=True,
    ):
        if not np.array_equal(original, copied):
            raise RuntimeError("materialized TIFFXYZ coordinate values changed")
    return {
        "directory": str(destination.resolve()),
        "role": role,
        "shape": list(mask.shape),
        "valid_vertex_count": int(np.count_nonzero(mask)),
        "active_quad_count": int(np.count_nonzero(selected_quads)),
        "physical_area_cm2": physical_area_from_mask(mask, quad_area_cm2),
        "coordinate_policy": "byte-exact source copies; explicit mask only",
        "no_bridging_proof": True,
        "files": files,
        "reload_manifest": reloaded.manifest(),
    }


def compact_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    positive = geometry["native_reports_by_orientation"]["positive"]
    return {
        "pass": bool(geometry["pass"]),
        "hard_defect_counts": geometry["hard_defect_counts"],
        "valid_vertex_count": int(positive["valid_vertex_count"]),
        "valid_quad_count": int(positive["valid_quad_count"]),
        "normal_valid_vertex_fraction": float(positive["normal_valid_vertex_fraction"]),
        "in_bounds_vertex_fraction": float(positive["in_bounds_vertex_fraction"]),
        "offset_sample_in_bounds_fraction": float(
            positive["offset_sample_in_bounds_fraction"]
        ),
        "self_intersection_status": geometry["self_intersection_status"],
        "positive_orientation": positive,
        "negative_orientation": geometry["native_reports_by_orientation"]["negative"],
    }


def fresh_surface_evaluation(
    surface: Any,
    sampler: PublicZarrChunkSampler,
    spec: ZarrV2ArraySpec,
) -> dict[str, Any]:
    """Recompute geometry, fresh normals, public-m7 runs, and centered quads."""

    masks, geometry = hard_geometry_summary(
        surface,
        volume_shape_zyx=spec.shape_zyx,
        voxel_um=VOXEL_UM,
    )
    distorted_margin = dilate_one_cell(mark_distorted_vertices(masks.distorted_quad))
    # Invalid canvas/hole vertices define an intentional mask boundary, not a
    # geometry defect to erode inward. Every valid vertex with a newly computed
    # one-sided normal and in-bounds transect is therefore sampled.
    geometry_eligible = surface.valid & masks.sample_valid
    frames, sample_validity = sample_offsets(
        sampler,
        surface,
        masks.normals_xyz,
        geometry_eligible,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(geometry_eligible, sample_validity.shape)
    sampling_complete = bool(np.all(sample_validity[requested]))
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset)
        for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support = support_summary(
        frames[support_indices], DEFAULT_SUPPORT_OFFSETS, geometry_eligible
    )
    selected_runs = selected_run_metrics(frames, geometry_eligible)
    centered = geometry_eligible & np.asarray(
        selected_runs["contains_zero"], dtype=bool
    )
    components, labels = connected_quad_components_summary(
        centered, masks.quad_area_cm2
    )
    largest = components["largest_component"]
    largest_mask = (
        quad_component_vertex_mask(
            labels,
            int(largest["component_id"]),
            surface.shape,
        )
        if largest is not None
        else np.zeros(surface.shape, dtype=bool)
    )
    no_bridging = bool(
        largest is not None
        and np.array_equal(active_quads(largest_mask), labels == largest["component_id"])
    )
    return {
        "masks": masks,
        "geometry": geometry,
        "distorted_margin": distorted_margin,
        "geometry_eligible": geometry_eligible,
        "frames": frames,
        "sample_validity": sample_validity,
        "sampling_complete": sampling_complete,
        "support": support,
        "selected_runs": selected_runs,
        "centered": centered,
        "components": components,
        "labels": labels,
        "largest": largest,
        "largest_mask": largest_mask,
        "no_bridging": no_bridging,
        "materialized_topology": topology_report(surface.valid),
        "largest_topology": topology_report(largest_mask)
        if largest is not None
        else None,
        "diagnostics": selected_region_diagnostics(selected_runs, largest_mask),
        "area_cm2": float(largest["area_cm2"]) if largest is not None else 0.0,
    }


def save_array(path: Path, array: NDArray[Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array), allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "content_sha256": _array_sha256(np.asarray(array)),
        "shape": list(np.asarray(array).shape),
        "dtype": np.asarray(array).dtype.str,
    }


def one_segment(report: Mapping[str, Any], long_id: str) -> Mapping[str, Any]:
    matches = [item for item in report["segments"] if item["long_id"] == long_id]
    if len(matches) != 1:
        raise ValueError(
            "preflight report does not contain exactly one fixed requested segment"
        )
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    long_id = str(args.long_id)
    expected_rank = int(args.expected_rank)
    preflight_path = Path(args.preflight_report)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "complete":
        raise ValueError("PHerc0800 preflight report is not complete")
    segment = one_segment(preflight, long_id)
    ranking = [
        row
        for row in preflight["all_segments_lexicographic_preflight_ranking"]
        if row["long_id"] == long_id
    ]
    if len(ranking) != 1 or int(ranking[0]["rank"]) != expected_rank:
        raise ValueError(
            f"fixed segment is no longer preflight rank {expected_rank}"
        )
    source_dir = Path(preflight_path.parent / "assets" / long_id)
    component_labels = np.load(
        segment["array_outputs"]["connected_quad_component_labels"]["path"],
        allow_pickle=False,
    )
    largest = segment["connected_quad_regions"]["largest_component"]
    selected_quad_mask = component_labels == int(largest["component_id"])
    selected_vertex_mask = quad_union_vertex_mask(selected_quad_mask)
    if not np.array_equal(active_quads(selected_vertex_mask), selected_quad_mask):
        raise ValueError(
            "preflight component cannot be encoded as a vertex mask without bridging"
        )
    source_asset = load_tifxyz_asset(source_dir, resolution="stored")
    source_masks, _source_geometry = hard_geometry_summary(
        source_asset.surface,
        volume_shape_zyx=preflight["m7"]["array_spec"]["shape_zyx"],
        voxel_um=VOXEL_UM,
    )
    output_dir = Path(args.output)
    materialized_dir = output_dir / "materialized-preflight-component"
    materialized = materialize_masked_tifxyz(
        source_dir,
        materialized_dir,
        selected_vertex_mask,
        quad_area_cm2=source_masks.quad_area_cm2,
        role=f"preflight_rank{expected_rank}_largest_centered_quad_component",
        lineage={
            "preflight_report": str(preflight_path.resolve()),
            "preflight_report_sha256": sha256_file(preflight_path),
            "source_long_id": long_id,
            "source_component_id": int(largest["component_id"]),
            "source_component_area_cm2": float(largest["area_cm2"]),
            "selection_fixed_before_fresh_boundary_normal_validation": True,
        },
    )
    materialized_asset = load_tifxyz_asset(materialized_dir, resolution="stored")
    # ZarrV2ArraySpec expects metadata keys rather than the compact report names.
    m7_array_metadata, array_provenance = fetch_json(
        str(preflight["m7"]["root"]).rstrip("/")
        + "/"
        + str(preflight["m7"]["array_path"]).strip("/")
        + "/.zarray"
    )
    spec = ZarrV2ArraySpec.from_metadata(m7_array_metadata)
    attributes, attrs_provenance = fetch_json(
        str(preflight["m7"]["root"]).rstrip("/") + "/.zattrs"
    )
    validate_metadata_axes(attributes, str(preflight["m7"]["array_path"]))
    payload_fetcher = DiskPayloadFetcher(Path(args.m7_cache))
    sampler = PublicZarrChunkSampler(
        root_url=str(preflight["m7"]["root"]),
        array_path=str(preflight["m7"]["array_path"]),
        spec=spec,
        blosc_path=Path(args.blosc),
        fetcher=payload_fetcher,
    )
    masks, geometry = hard_geometry_summary(
        materialized_asset.surface,
        volume_shape_zyx=spec.shape_zyx,
        voxel_um=VOXEL_UM,
    )
    distorted_margin = dilate_one_cell(mark_distorted_vertices(masks.distorted_quad))
    # Invalid canvas/hole vertices define the intentional mask boundary; they are
    # not defects to dilate inward. Native whole-mask geometry remains the hard
    # defect gate, while every valid vertex with a fresh valid normal/bounds
    # transect is sampled—including one-sided boundary and hole normals.
    geometry_eligible = materialized_asset.surface.valid & masks.sample_valid
    frames, sample_validity = sample_offsets(
        sampler,
        materialized_asset.surface,
        masks.normals_xyz,
        geometry_eligible,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(geometry_eligible, sample_validity.shape)
    sampling_complete = bool(np.all(sample_validity[requested]))
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset) for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support = support_summary(
        frames[support_indices], DEFAULT_SUPPORT_OFFSETS, geometry_eligible
    )
    selected_runs = selected_run_metrics(frames, geometry_eligible)
    fresh_centered = geometry_eligible & np.asarray(
        selected_runs["contains_zero"], dtype=bool
    )
    fresh_components, fresh_labels = connected_quad_components_summary(
        fresh_centered, masks.quad_area_cm2
    )
    fresh_largest = fresh_components["largest_component"]
    fresh_mask = (
        quad_component_vertex_mask(
            fresh_labels,
            int(fresh_largest["component_id"]),
            materialized_asset.surface.shape,
        )
        if fresh_largest is not None
        else np.zeros(materialized_asset.surface.shape, dtype=bool)
    )
    no_fresh_bridging = bool(
        fresh_largest is not None
        and np.array_equal(
            active_quads(fresh_mask), fresh_labels == fresh_largest["component_id"]
        )
    )
    original_topology = topology_report(materialized_asset.surface.valid)
    fresh_topology = topology_report(fresh_mask) if fresh_largest is not None else None
    calibration = preflight["published_surface_calibration_control"]["recomputed"]
    fresh_diagnostics = selected_region_diagnostics(selected_runs, fresh_mask)
    fresh_area = float(fresh_largest["area_cm2"]) if fresh_largest is not None else 0.0
    repaired_mask, repair = conservative_manifold_subset(
        fresh_mask, masks.quad_area_cm2
    )
    repaired_dir = output_dir / "materialized-topology-repaired-component"
    repaired_materialized = materialize_masked_tifxyz(
        source_dir,
        repaired_dir,
        repaired_mask,
        quad_area_cm2=masks.quad_area_cm2,
        role="fresh_centered_topology_repaired_render_component",
        lineage={
            "validated_materialized_component": str(materialized_dir.resolve()),
            "pre_repair_fresh_mask_content_sha256": _array_sha256(fresh_mask),
            "repair_algorithm": repair["algorithm"],
            "repair_iteration_count": int(repair["iteration_count"]),
            "repair_is_strictly_subtractive": True,
            "fixed_coordinates": True,
            "fresh_boundary_and_hole_normals_required_after_repair": True,
            "no_per_vertex_retuning": True,
        },
    )
    repaired_asset = load_tifxyz_asset(repaired_dir, resolution="stored")
    post_repair = fresh_surface_evaluation(repaired_asset.surface, sampler, spec)
    post_mask_identical = bool(
        np.array_equal(post_repair["largest_mask"], repaired_mask)
    )
    post_support_fraction = float(post_repair["support"]["support_fraction"])
    gate_results = [
        {
            "name": "preflight_component_vertex_mask_encodes_exact_quads_without_bridging",
            "pass": True,
        },
        {"name": "pre_repair_fresh_native_geometry", "pass": bool(geometry["pass"])},
        {
            "name": "pre_repair_fresh_centered_area",
            "value_cm2": fresh_area,
            "threshold_cm2": float(args.minimum_area_cm2),
            "pass": fresh_area >= float(args.minimum_area_cm2),
        },
        {
            "name": "deterministic_repair_is_strictly_subtractive_and_idempotent",
            "pass": bool(
                repair["strict_quad_subset_of_input"] and repair["idempotent"]
            ),
        },
        {
            "name": "post_repair_fresh_native_geometry",
            "pass": bool(post_repair["geometry"]["pass"]),
        },
        {
            "name": "post_repair_materialized_topology_has_no_nonmanifold_risk",
            "pass": not bool(
                post_repair["materialized_topology"]["nonmanifold_risk"]
            ),
        },
        {
            "name": "post_repair_all_requested_public_m7_samples_valid",
            "pass": bool(post_repair["sampling_complete"]),
        },
        {
            "name": "post_repair_plus_minus2_support",
            "value_fraction": post_support_fraction,
            "threshold_fraction": 0.9,
            "pass": post_support_fraction >= 0.9,
        },
        {
            "name": "post_repair_fresh_centered_connected_quad_area",
            "value_cm2": float(post_repair["area_cm2"]),
            "threshold_cm2": float(args.minimum_area_cm2),
            "pass": float(post_repair["area_cm2"])
            >= float(args.minimum_area_cm2),
        },
        {
            "name": "post_repair_fresh_passing_component_encodable_without_bridging",
            "pass": bool(post_repair["no_bridging"]),
        },
        {
            "name": "post_repair_fresh_passing_component_topology_has_no_nonmanifold_risk",
            "pass": bool(
                post_repair["largest_topology"] is not None
                and not post_repair["largest_topology"]["nonmanifold_risk"]
            ),
        },
        {
            "name": "post_repair_fresh_centered_mask_equals_fixed_repaired_mask",
            "pass": post_mask_identical,
            "reason": (
                "prevents an adaptive second crop after fresh repaired-boundary normals"
            ),
        },
    ]
    pass_to_raw = all(bool(gate["pass"]) for gate in gate_results)
    render_materialized = repaired_materialized if pass_to_raw else None
    arrays = {
        "preflight_selected_quad_mask": save_array(
            output_dir / "arrays/preflight-selected-quad-mask.npy", selected_quad_mask
        ),
        "materialized_vertex_mask": save_array(
            output_dir / "arrays/materialized-vertex-mask.npy", selected_vertex_mask
        ),
        "fresh_normals_xyz": save_array(
            output_dir / "arrays/fresh-normals-xyz.npy", masks.normals_xyz
        ),
        "geometry_eligible_mask": save_array(
            output_dir / "arrays/geometry-eligible-mask.npy", geometry_eligible
        ),
        "m7_offsets_minus15_plus15": save_array(
            output_dir / "arrays/m7-offsets-minus15-plus15.npy", frames
        ),
        "m7_sample_validity": save_array(
            output_dir / "arrays/m7-sample-validity.npy", sample_validity
        ),
        "selected_run_center": save_array(
            output_dir / "arrays/selected-run-center.npy",
            selected_runs["selected_center_offset"],
        ),
        "competitor_empty_gap": save_array(
            output_dir / "arrays/competitor-empty-gap.npy",
            selected_runs["nearest_competitor_empty_gap"],
        ),
        "fresh_centered_vertex_mask": save_array(
            output_dir / "arrays/fresh-centered-vertex-mask.npy", fresh_centered
        ),
        "fresh_quad_component_labels": save_array(
            output_dir / "arrays/fresh-quad-component-labels.npy", fresh_labels
        ),
        "fresh_largest_component_vertex_mask": save_array(
            output_dir / "arrays/fresh-largest-component-vertex-mask.npy", fresh_mask
        ),
        "quad_area_cm2": save_array(
            output_dir / "arrays/quad-area-cm2.npy", masks.quad_area_cm2
        ),
        "topology_repaired_vertex_mask": save_array(
            output_dir / "arrays/topology-repaired-vertex-mask.npy", repaired_mask
        ),
        "post_repair_fresh_normals_xyz": save_array(
            output_dir / "arrays/post-repair-fresh-normals-xyz.npy",
            post_repair["masks"].normals_xyz,
        ),
        "post_repair_geometry_eligible_mask": save_array(
            output_dir / "arrays/post-repair-geometry-eligible-mask.npy",
            post_repair["geometry_eligible"],
        ),
        "post_repair_m7_offsets_minus15_plus15": save_array(
            output_dir / "arrays/post-repair-m7-offsets-minus15-plus15.npy",
            post_repair["frames"],
        ),
        "post_repair_m7_sample_validity": save_array(
            output_dir / "arrays/post-repair-m7-sample-validity.npy",
            post_repair["sample_validity"],
        ),
        "post_repair_selected_run_center": save_array(
            output_dir / "arrays/post-repair-selected-run-center.npy",
            post_repair["selected_runs"]["selected_center_offset"],
        ),
        "post_repair_competitor_empty_gap": save_array(
            output_dir / "arrays/post-repair-competitor-empty-gap.npy",
            post_repair["selected_runs"]["nearest_competitor_empty_gap"],
        ),
        "post_repair_fresh_centered_vertex_mask": save_array(
            output_dir / "arrays/post-repair-fresh-centered-vertex-mask.npy",
            post_repair["centered"],
        ),
        "post_repair_quad_component_labels": save_array(
            output_dir / "arrays/post-repair-quad-component-labels.npy",
            post_repair["labels"],
        ),
        "post_repair_largest_component_vertex_mask": save_array(
            output_dir / "arrays/post-repair-largest-component-vertex-mask.npy",
            post_repair["largest_mask"],
        ),
        "post_repair_quad_area_cm2": save_array(
            output_dir / "arrays/post-repair-quad-area-cm2.npy",
            post_repair["masks"].quad_area_cm2,
        ),
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "candidate_id": long_id,
        "verdict": "pass_to_free_raw_render"
        if pass_to_raw
        else "stop_before_raw_render",
        "paid_compute_cost_usd": 0.0,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "source_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": sha256_file(preflight_path),
            "rank": expected_rank,
            "fixed_component": largest,
        },
        "source_tifxyz": source_asset.manifest(),
        "materialized_preflight_component": materialized,
        "pre_repair_fresh_centered_component": {
            "area_cm2": fresh_area,
            "regions": fresh_components,
            "diagnostics": fresh_diagnostics,
            "topology": fresh_topology,
            "vertex_mask_content_sha256": _array_sha256(fresh_mask),
        },
        "deterministic_topology_repair": repair,
        "materialized_topology_repaired_component": repaired_materialized,
        "fresh_native_geometry": compact_geometry(geometry),
        "fresh_localized_geometry": {
            **masks.summary,
            "geometry_eligible_vertex_count": int(np.count_nonzero(geometry_eligible)),
            "distorted_quad_vertex_margin_count": int(
                np.count_nonzero(distorted_margin)
            ),
            "intentional_mask_boundary_policy": (
                "invalid canvas/hole vertices are not dilated inward; native whole-mask "
                "geometry is the hard defect gate and fresh one-sided boundary normals "
                "are explicitly sampled"
            ),
        },
        "topology_preflight_component_after_materialization": original_topology,
        "m7": {
            "root": str(preflight["m7"]["root"]),
            "array_path": str(preflight["m7"]["array_path"]),
            "array_metadata": array_provenance,
            "root_attributes": attrs_provenance,
            "codec_path": str(Path(args.blosc).resolve()),
            "codec_sha256": sha256_file(Path(args.blosc)),
            "sampler": sampler.manifest(),
            "payload_cache": payload_fetcher.manifest(),
            "sampling_complete": sampling_complete,
            "plus_minus2_support": support,
            "selected_run_metrics": run_metrics_summary(
                selected_runs, geometry_eligible
            ),
            "calibration_policy": (
                "hard gate is a centered selected run on every vertex counted in the "
                "passing component; secondary runs and clearance are non-veto diagnostics"
            ),
            "published_control_reference": calibration,
        },
        "fresh_centered_connected_quad_regions": fresh_components,
        "fresh_largest_component_diagnostics": fresh_diagnostics,
        "fresh_largest_component_topology": fresh_topology,
        "post_repair_fresh_validation": {
            "native_geometry": compact_geometry(post_repair["geometry"]),
            "localized_geometry": {
                **post_repair["masks"].summary,
                "geometry_eligible_vertex_count": int(
                    np.count_nonzero(post_repair["geometry_eligible"])
                ),
                "distorted_quad_vertex_margin_count": int(
                    np.count_nonzero(post_repair["distorted_margin"])
                ),
                "intentional_mask_boundary_policy": (
                    "invalid canvas/hole vertices are not dilated inward; native "
                    "whole-mask geometry is the hard defect gate and newly recomputed "
                    "one-sided repair-boundary normals are explicitly sampled"
                ),
            },
            "materialized_mask_topology": post_repair["materialized_topology"],
            "sampling_complete": bool(post_repair["sampling_complete"]),
            "plus_minus2_support": post_repair["support"],
            "selected_run_metrics": run_metrics_summary(
                post_repair["selected_runs"], post_repair["geometry_eligible"]
            ),
            "fresh_centered_connected_quad_regions": post_repair["components"],
            "fresh_largest_component_diagnostics": post_repair["diagnostics"],
            "fresh_largest_component_topology": post_repair["largest_topology"],
            "fresh_largest_mask_identical_to_repaired_materialized_mask": (
                post_mask_identical
            ),
        },
        "calibration_comparison": {
            "candidate_selected_center_absolute": post_repair["diagnostics"][
                "selected_run_center_absolute"
            ],
            "control_selected_center_absolute_all_selected": calibration[
                "selected_run_metrics"
            ]["selected_run_center_absolute_all_selected"],
            "candidate_competitor_empty_gap": post_repair["diagnostics"][
                "nearest_competitor_empty_gap"
            ],
            "control_competitor_empty_gap_all_selected": calibration[
                "selected_run_metrics"
            ]["nearest_competitor_empty_gap_all_selected"],
            "clearance_gate_role": "diagnostic comparison only; no invented hard threshold",
        },
        "gate_results": gate_results,
        "pass_to_raw_render": pass_to_raw,
        "materialized_fresh_passing_render_component": render_materialized,
        "array_outputs": arrays,
        "limitations": [
            "Self-intersection was not computed and remains unknown.",
            "Public m7 is a surface-prediction reference, not independent raw-CT proof.",
            "Holes are preserved; no coordinate was moved, clamped, filled, or bridged.",
            "Clearance is calibrated and reported but is not an unvalidated hard veto.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-id", default=LONG_ID)
    parser.add_argument("--expected-rank", type=int, default=2)
    parser.add_argument(
        "--preflight-report", type=Path, default=DEFAULT_PREFLIGHT_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--m7-cache", type=Path, default=DEFAULT_PREFLIGHT_DIR / "m7-chunk-cache"
    )
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument(
        "--minimum-area-cm2", type=float, default=DEFAULT_MINIMUM_AREA_CM2
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output) / "validation.json"
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema_version": 2,
            "status": "error",
            "candidate_id": str(args.long_id),
            "verdict": "validation_error",
            "paid_compute_cost_usd": 0.0,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(output_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    _atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "fresh_area_cm2": result["post_repair_fresh_validation"][
                    "fresh_centered_connected_quad_regions"
                ]["largest_component"]["area_cm2"],
                "topology": result["post_repair_fresh_validation"][
                    "fresh_largest_component_topology"
                ],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
