#!/usr/bin/env python3
"""Archived surgical ragged-growth helpers from the pre-calibration v3 probe.

The original executable treated exactly one foreground run across +/-15 as a
hard sheet-identity gate. A clean published calibration surface falsified that
assumption, so :func:`main` now fails closed before loading inputs. The
topology, geometry, and mask helpers remain importable by the calibrated v3.3
implementation; this module must not be used to produce scientific results.

No raw CT, ink inference, detector, paid GPU, or RunPod is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from m7_locked_directional_grower import (
    DEFAULT_M7_ROOT,
    DEFAULT_VOXEL_UM,
    canonical_trilinear_transect,
    load_public_sampler,
)
from m7_locked_surface_grower import (
    GrowConfig,
    PublicM7BinaryAccessor,
    _array_sha256,
    _json_safe,
    project_target_to_m7,
    transported_basis_zyx,
)
from native_surface_sampler import (
    SurfaceGrid,
    estimate_surface_normals,
    validate_surface_geometry,
)
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    PublicZarrChunkSampler,
    evaluate_all_vertex_transects,
    evaluate_valid_vertex_transects,
    support_summary,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-locked-surgical-ragged-v3.0"
ABORTED_SCIENTIFIC_GATE = True
DEFAULT_V2 = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v2-directional-0p05cm2-publicrun"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v3-surgical-ragged-0p05cm2"
)
DEFAULT_SEED_XYZ = (3525.0, 4191.0, 14071.0)


@dataclass(frozen=True)
class RaggedConfig:
    minimum_bootstrap_vertices: int = 50
    maximum_area_cm2: float = 0.05
    maximum_cycles: int = 256
    minimum_support_fraction: float = 0.90

    def validate(self) -> None:
        if self.minimum_bootstrap_vertices < 4:
            raise ValueError("minimum_bootstrap_vertices must be at least four")
        if self.maximum_area_cm2 <= 0:
            raise ValueError("maximum_area_cm2 must be positive")
        if self.maximum_cycles < 1:
            raise ValueError("maximum_cycles must be positive")
        if not 0 <= self.minimum_support_fraction <= 1:
            raise ValueError("minimum_support_fraction must be in [0,1]")


@dataclass(frozen=True)
class V2Input:
    points_xyz: NDArray[np.float64]
    generations: NDArray[np.uint16]
    logical_bounds: tuple[int, int, int, int]
    seed_stored_row_column: tuple[int, int]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class SurfaceEvaluation:
    passed: bool
    area_cm2: float
    surface: SurfaceGrid
    normals_positive_xyz: NDArray[np.float64]
    native_reports: Mapping[str, Any]
    support: Mapping[str, Any]
    exhaustive: Mapping[str, Any]
    coherence: Mapping[str, Any]
    sampling_complete: bool
    frames_sha256: str
    sample_valid_sha256: str


@dataclass(frozen=True)
class BootstrapResult:
    rows: tuple[int, int]
    columns: tuple[int, int]
    points: Mapping[tuple[int, int], tuple[float, float, float]]
    generations: Mapping[tuple[int, int], int]
    active_quads: frozenset[tuple[int, int]]
    evaluation: SurfaceEvaluation
    candidate_count: int
    passing_candidate_count: int
    candidate_summaries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RaggedState:
    points_zyx: Mapping[tuple[int, int], tuple[float, float, float]]
    generations: Mapping[tuple[int, int], int]
    active_quads: frozenset[tuple[int, int]]
    logical_seed: tuple[int, int]
    evaluation: SurfaceEvaluation


@dataclass(frozen=True)
class QuadTrial:
    quad: tuple[int, int]
    passed: bool
    reason: str
    new_vertex_records: tuple[Mapping[str, Any], ...]
    state: RaggedState | None
    topology: Mapping[str, Any] | None
    area_gain_cm2: float | None


def _read_manifest(directory: Path) -> Mapping[str, str]:
    path = directory / "MANIFEST.sha256"
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        member = directory / name
        if not member.is_file() or sha256_file(member) != digest:
            raise ValueError(f"v2 manifest mismatch: {name}")
        hashes[name] = digest
    return hashes


def load_v2_input(directory: Path) -> V2Input:
    directory = directory.resolve()
    hashes = _read_manifest(directory)
    meta = json.loads((directory / "meta.json").read_text())
    provenance = json.loads((directory / "growth-provenance.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if provenance.get("algorithm_version") != "m7-locked-directional-edges-v2.0":
        raise ValueError("input is not the pinned v2 directional artifact")
    if provenance.get("status") != "complete":
        raise ValueError("v2 input is incomplete")
    if not all(
        gate.get("pass")
        for gate in validation.get("gates", [])
        if gate.get("name") in {
            "minimum_50_contiguous_rectangle_vertices",
            "native_geometry_both_orientations",
            "all_requested_m7_samples_valid",
            "five_offset_m7_support",
        }
    ):
        raise ValueError("v2 prerequisite geometry/support gates do not pass")
    asset = load_tifxyz_asset(directory, resolution="stored")
    points_xyz = asset.surface.points_xyz
    if not bool(asset.surface.valid.all()):
        raise ValueError("v2 base unexpectedly contains masked vertices")
    generations = np.asarray(tifffile.imread(directory / "generations.tif"), dtype=np.uint16)
    if generations.shape != points_xyz.shape[:2]:
        raise ValueError("v2 generation shape mismatch")
    bounds = tuple(int(value) for value in meta["logical_bounds"])
    r0, r1, c0, c1 = bounds
    if points_xyz.shape[:2] != (r1 - r0 + 1, c1 - c0 + 1):
        raise ValueError("v2 logical bounds disagree with stored shape")
    seed_logical = (0, 0)
    seed_stored = (seed_logical[0] - r0, seed_logical[1] - c0)
    return V2Input(
        points_xyz=points_xyz,
        generations=generations,
        logical_bounds=bounds,
        seed_stored_row_column=seed_stored,
        evidence={
            "directory": str(directory),
            "manifest_sha256": sha256_file(directory / "MANIFEST.sha256"),
            "member_sha256": hashes,
            "stored_asset_hashes": asset.hashes,
            "validation_sha256": sha256_file(directory / "validation.json"),
            "growth_provenance_sha256": sha256_file(directory / "growth-provenance.json"),
        },
    )


def induced_quads(vertices: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    vertex_set = set(vertices)
    if not vertex_set:
        return frozenset()
    r_values = [key[0] for key in vertex_set]
    c_values = [key[1] for key in vertex_set]
    quads = {
        (row, column)
        for row in range(min(r_values), max(r_values))
        for column in range(min(c_values), max(c_values))
        if {
            (row, column),
            (row + 1, column),
            (row, column + 1),
            (row + 1, column + 1),
        }.issubset(vertex_set)
    }
    return frozenset(quads)


def quad_vertices(quad: tuple[int, int]) -> frozenset[tuple[int, int]]:
    row, column = quad
    return frozenset(
        ((row, column), (row + 1, column), (row, column + 1), (row + 1, column + 1))
    )


def vertices_for_quads(quads: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    vertices: set[tuple[int, int]] = set()
    for quad in quads:
        vertices.update(quad_vertices(quad))
    return frozenset(vertices)


def topology_audit(
    quads: Iterable[tuple[int, int]], *, seed: tuple[int, int]
) -> Mapping[str, Any]:
    """Certify that an induced unit-quad complex is one topological disk."""

    quad_set = set(quads)
    vertices = vertices_for_quads(quad_set)
    if not quad_set or seed not in vertices:
        return {"pass": False, "reason": "empty_or_seed_absent"}
    # Quad edge connectivity.
    pending = [min(quad_set)]
    reached: set[tuple[int, int]] = set()
    while pending:
        quad = pending.pop()
        if quad in reached:
            continue
        reached.add(quad)
        row, column = quad
        for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if neighbor in quad_set and neighbor not in reached:
                pending.append(neighbor)

    edge_incidence: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for row, column in quad_set:
        corners = ((row, column), (row, column + 1), (row + 1, column + 1), (row + 1, column))
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = tuple(sorted((first, second)))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
    nonmanifold_edges = [edge for edge, count in edge_incidence.items() if count not in (1, 2)]
    boundary_edges = [edge for edge, count in edge_incidence.items() if count == 1]
    boundary_degree: dict[tuple[int, int], int] = {}
    for first, second in boundary_edges:
        boundary_degree[first] = boundary_degree.get(first, 0) + 1
        boundary_degree[second] = boundary_degree.get(second, 0) + 1
    boundary_degree_bad = [vertex for vertex, degree in boundary_degree.items() if degree != 2]
    boundary_components = 0
    unseen = set(boundary_degree)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in boundary_degree}
    for first, second in boundary_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    while unseen:
        boundary_components += 1
        stack = [min(unseen)]
        while stack:
            vertex = stack.pop()
            if vertex not in unseen:
                continue
            unseen.remove(vertex)
            stack.extend(adjacency[vertex] & unseen)
    euler = len(vertices) - len(edge_incidence) + len(quad_set)
    induced = induced_quads(vertices)
    passed = bool(
        reached == quad_set
        and not nonmanifold_edges
        and not boundary_degree_bad
        and boundary_components == 1
        and euler == 1
        and induced == frozenset(quad_set)
    )
    return {
        "pass": passed,
        "reason": "pass" if passed else "topology_gate_failed",
        "quad_count": len(quad_set),
        "vertex_count": len(vertices),
        "edge_count": len(edge_incidence),
        "quad_component_count": 1 if reached == quad_set else 2,
        "boundary_component_count": boundary_components,
        "boundary_edge_count": len(boundary_edges),
        "boundary_degree_failure_count": len(boundary_degree_bad),
        "nonmanifold_edge_count": len(nonmanifold_edges),
        "euler_characteristic": euler,
        "induced_quad_set_exact": induced == frozenset(quad_set),
        "seed_active": seed in vertices,
    }


def grid_from_points(
    points: Mapping[tuple[int, int], Sequence[float]],
    quads: Iterable[tuple[int, int]],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], tuple[int, int, int, int]]:
    active = vertices_for_quads(quads)
    if active != frozenset(points):
        raise ValueError("point keys must equal vertices incident to active quads")
    rows = [key[0] for key in active]
    columns = [key[1] for key in active]
    bounds = (min(rows), max(rows), min(columns), max(columns))
    r0, r1, c0, c1 = bounds
    grid = np.full((r1 - r0 + 1, c1 - c0 + 1, 3), np.nan, dtype=np.float64)
    mask = np.zeros(grid.shape[:2], dtype=bool)
    for (row, column), point in points.items():
        grid[row - r0, column - c0] = point
        mask[row - r0, column - c0] = True
    return grid, mask, bounds


def masked_area_cm2(
    points_zyx: Mapping[tuple[int, int], Sequence[float]],
    quads: Iterable[tuple[int, int]],
    voxel_um: float,
) -> float:
    area_um2 = 0.0
    for row, column in quads:
        p00 = np.asarray(points_zyx[(row, column)]) * voxel_um
        p01 = np.asarray(points_zyx[(row, column + 1)]) * voxel_um
        p10 = np.asarray(points_zyx[(row + 1, column)]) * voxel_um
        p11 = np.asarray(points_zyx[(row + 1, column + 1)]) * voxel_um
        area_um2 += 0.5 * (
            np.linalg.norm(np.cross(p01 - p00, p10 - p00))
            + np.linalg.norm(np.cross(p11 - p10, p11 - p01))
        )
    return float(area_um2 / 100_000_000.0)


def ragged_geometry_pass(report: Mapping[str, Any], *, zero_distortion: bool) -> bool:
    zero_keys = [
        "discontinuity_edge_count",
        "neighbor_wrap_risk_edge_count",
        "degenerate_quad_count",
        "folded_quad_count",
        "abrupt_normal_flip_edge_count",
    ]
    if zero_distortion:
        zero_keys.append("distorted_quad_count")
    return bool(
        all(int(report[key]) == 0 for key in zero_keys)
        and float(report["in_bounds_vertex_fraction"]) == 1.0
        and float(report["normal_valid_vertex_fraction"]) == 1.0
        # For an intentionally masked ragged grid the native denominator is
        # every edge in the rectangular bounding box, including edges touching
        # masked cells.  The campaign's explicit mandatory gate is >=0.90;
        # requiring 1.0 would make every legitimate boundary indentation fail.
        and float(report["valid_neighbor_edge_fraction"]) >= 0.90
        and float(report["offset_sample_in_bounds_fraction"]) == 1.0
        and float(report["distorted_quad_fraction"]) <= 0.05
    )


def coherence_report(
    points_zyx: Mapping[tuple[int, int], Sequence[float]],
    quads: Iterable[tuple[int, int]],
    surface: SurfaceGrid,
    normals_positive_xyz: NDArray[np.float64],
    bounds: Sequence[int],
    grow_config: GrowConfig,
) -> Mapping[str, Any]:
    r0, _, c0, _ = (int(value) for value in bounds)
    active_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row, column in quads:
        for first, second in (
            ((row, column), (row, column + 1)),
            ((row, column), (row + 1, column)),
            ((row + 1, column), (row + 1, column + 1)),
            ((row, column + 1), (row + 1, column + 1)),
        ):
            active_edges.add(tuple(sorted((first, second))))
    failures: list[Mapping[str, Any]] = []
    minimum_dot = 1.0
    maximum_distance_error = 0.0
    maximum_normal_fraction = 0.0
    for first, second in sorted(active_edges):
        first_stored = (first[0] - r0, first[1] - c0)
        second_stored = (second[0] - r0, second[1] - c0)
        first_point = np.asarray(points_zyx[first], dtype=np.float64)
        second_point = np.asarray(points_zyx[second], dtype=np.float64)
        first_normal = np.asarray(normals_positive_xyz[first_stored], dtype=np.float64)[::-1]
        second_normal = np.asarray(normals_positive_xyz[second_stored], dtype=np.float64)[::-1]
        dot = abs(float(first_normal @ second_normal))
        displacement = second_point - first_point
        distance = float(np.linalg.norm(displacement))
        distance_error = abs(distance - grow_config.spacing_voxels) / grow_config.spacing_voxels
        average_normal = first_normal + second_normal
        average_normal /= np.linalg.norm(average_normal)
        normal_fraction = abs(float(displacement @ average_normal)) / max(distance, 1e-12)
        minimum_dot = min(minimum_dot, dot)
        maximum_distance_error = max(maximum_distance_error, distance_error)
        maximum_normal_fraction = max(maximum_normal_fraction, normal_fraction)
        if (
            dot + 1e-12 < grow_config.minimum_neighbor_normal_dot
            or distance_error > grow_config.neighbor_distance_relative_tolerance
            or normal_fraction > grow_config.maximum_neighbor_normal_displacement_fraction
        ):
            failures.append(
                {"first": list(first), "second": list(second), "normal_abs_dot": dot,
                 "distance_relative_error": distance_error,
                 "normal_displacement_fraction": normal_fraction}
            )
    return {
        "pass": not failures,
        "active_edge_count": len(active_edges),
        "minimum_neighbor_normal_abs_dot": minimum_dot,
        "maximum_neighbor_distance_relative_error": maximum_distance_error,
        "maximum_neighbor_normal_displacement_fraction": maximum_normal_fraction,
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def evaluate_surface(
    points_zyx: Mapping[tuple[int, int], Sequence[float]],
    quads: Iterable[tuple[int, int]],
    sampler: PublicZarrChunkSampler,
    config: RaggedConfig,
    grow_config: GrowConfig,
    *,
    voxel_um: float,
    require_zero_distortion: bool,
) -> SurfaceEvaluation:
    grid, mask, bounds = grid_from_points(points_zyx, quads)
    xyz = grid[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(
        xyz[..., 0], xyz[..., 1], xyz[..., 2], mask=mask
    )
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    offsets_um = [value * voxel_um for value in DEFAULT_TRANSECT_OFFSETS]
    reports = {
        label: validate_surface_geometry(
            surface,
            volume_shape_zyx=sampler.shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }
    geometry_pass = all(
        ragged_geometry_pass(report, zero_distortion=require_zero_distortion)
        for report in reports.values()
    )
    frames, sample_valid = sample_offsets(
        sampler,
        surface,
        normals.positive_xyz,
        surface.valid & normals.valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested_valid = np.broadcast_to(surface.valid, sample_valid.shape)
    sampling_complete = bool(np.all(sample_valid[requested_valid]))
    support_indices = [DEFAULT_TRANSECT_OFFSETS.index(value) for value in DEFAULT_SUPPORT_OFFSETS]
    support = support_summary(frames[support_indices], DEFAULT_SUPPORT_OFFSETS, surface.valid)
    exhaustive = evaluate_all_vertex_transects(frames, surface.valid)
    coherence = coherence_report(
        points_zyx, quads, surface, normals.positive_xyz, bounds, grow_config
    )
    area = masked_area_cm2(points_zyx, quads, voxel_um)
    passed = bool(
        geometry_pass
        and sampling_complete
        and float(support["support_fraction"]) >= config.minimum_support_fraction
        and exhaustive.get("all_clean", False)
        and coherence["pass"]
    )
    return SurfaceEvaluation(
        passed=passed,
        area_cm2=area,
        surface=surface,
        normals_positive_xyz=normals.positive_xyz,
        native_reports=reports,
        support=support,
        exhaustive=exhaustive,
        coherence=coherence,
        sampling_complete=sampling_complete,
        frames_sha256=_array_sha256(frames),
        sample_valid_sha256=_array_sha256(sample_valid),
    )


def enumerate_clean_bootstrap(
    source: V2Input,
    sampler: PublicZarrChunkSampler,
    config: RaggedConfig,
    *,
    voxel_um: float,
) -> BootstrapResult:
    """Exhaustively test all seed-containing stored-grid rectangles >=50."""

    height, width = source.points_xyz.shape[:2]
    seed_row, seed_column = source.seed_stored_row_column
    rbase, _, cbase, _ = source.logical_bounds
    candidates: list[tuple[float, int, tuple[int, int, int, int], SurfaceEvaluation, dict, dict, frozenset]] = []
    summaries: list[Mapping[str, Any]] = []
    evaluated_count = 0
    for row_start in range(seed_row + 1):
        for row_stop in range(seed_row + 1, height + 1):
            for column_start in range(seed_column + 1):
                for column_stop in range(seed_column + 1, width + 1):
                    vertex_count = (row_stop - row_start) * (column_stop - column_start)
                    if vertex_count < config.minimum_bootstrap_vertices:
                        continue
                    evaluated_count += 1
                    points: dict[tuple[int, int], tuple[float, float, float]] = {}
                    generations: dict[tuple[int, int], int] = {}
                    for stored_row in range(row_start, row_stop):
                        for stored_column in range(column_start, column_stop):
                            logical = (rbase + stored_row, cbase + stored_column)
                            points[logical] = tuple(
                                float(value)
                                for value in source.points_xyz[stored_row, stored_column, ::-1]
                            )
                            generations[logical] = int(source.generations[stored_row, stored_column])
                    quads = induced_quads(points)
                    topology = topology_audit(quads, seed=(0, 0))
                    if not topology["pass"]:
                        continue
                    evaluation = evaluate_surface(
                        points, quads, sampler, config,
                        voxel_um=voxel_um, require_zero_distortion=True,
                    )
                    summary = {
                        "stored_rows_half_open": [row_start, row_stop],
                        "stored_columns_half_open": [column_start, column_stop],
                        "shape": [row_stop - row_start, column_stop - column_start],
                        "vertex_count": vertex_count,
                        "area_cm2": evaluation.area_cm2,
                        "pass": evaluation.passed,
                        "geometry_pass": all(
                            ragged_geometry_pass(report, zero_distortion=True)
                            for report in evaluation.native_reports.values()
                        ),
                        "support_fraction": evaluation.support["support_fraction"],
                        "all_vertex_category_counts": evaluation.exhaustive["category_counts"],
                    }
                    summaries.append(summary)
                    if evaluation.passed:
                        candidates.append(
                            (
                                -evaluation.area_cm2,
                                -vertex_count,
                                (row_start, row_stop, column_start, column_stop),
                                evaluation,
                                points,
                                generations,
                                quads,
                            )
                        )
    if not candidates:
        raise RuntimeError("no seed-containing >=50 clean rectangular bootstrap exists")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, box, evaluation, points, generations, quads = candidates[0]
    row_start, row_stop, column_start, column_stop = box
    # Keep all summaries compact but sufficient to prove exhaustive coverage.
    compact_summaries = tuple(
        sorted(
            summaries,
            key=lambda item: (
                not bool(item["pass"]),
                -float(item["area_cm2"]),
                item["stored_rows_half_open"],
                item["stored_columns_half_open"],
            ),
        )
    )
    return BootstrapResult(
        rows=(row_start, row_stop),
        columns=(column_start, column_stop),
        points=points,
        generations=generations,
        active_quads=quads,
        evaluation=evaluation,
        candidate_count=evaluated_count,
        passing_candidate_count=len(candidates),
        candidate_summaries=compact_summaries,
    )


def frontier_quads(active_quads: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    active = set(active_quads)
    frontier: set[tuple[int, int]] = set()
    for row, column in active:
        for candidate in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if candidate not in active:
                frontier.add(candidate)
    return tuple(sorted(frontier))


def _normal_map(state: RaggedState) -> Mapping[tuple[int, int], NDArray[np.float64]]:
    _, mask, bounds = grid_from_points(state.points_zyx, state.active_quads)
    r0, _, c0, _ = bounds
    normals_xyz = state.evaluation.normals_positive_xyz
    result: dict[tuple[int, int], NDArray[np.float64]] = {}
    for row, column in state.points_zyx:
        stored = (row - r0, column - c0)
        if not mask[stored]:
            raise ValueError("active point absent from evaluation mask")
        result[(row, column)] = np.asarray(normals_xyz[stored])[::-1]
    return result


def _reference_basis(state: RaggedState) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normals = _normal_map(state)
    normal = normals[state.logical_seed]
    normal /= np.linalg.norm(normal)
    reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
    column = np.cross(normal, reference)
    column /= np.linalg.norm(column)
    pivot = int(np.argmax(np.abs(column)))
    if column[pivot] < 0:
        column = -column
    row = np.cross(normal, column)
    row /= np.linalg.norm(row)
    return row, column


def _propose_new_vertex(
    key: tuple[int, int],
    state: RaggedState,
    normals: Mapping[tuple[int, int], NDArray[np.float64]],
    reference_row: Sequence[float],
    reference_column: Sequence[float],
    grow_config: GrowConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    neighbors = [
        neighbor
        for neighbor in sorted(state.points_zyx)
        if max(abs(neighbor[0] - key[0]), abs(neighbor[1] - key[1])) <= 1
    ]
    if not neighbors:
        raise ValueError("frontier vertex has no active neighbor")
    proposals: list[NDArray[np.float64]] = []
    neighbor_normals: list[NDArray[np.float64]] = []
    for neighbor in neighbors:
        normal = normals[neighbor]
        local_row, local_column = transported_basis_zyx(
            normal, reference_row, reference_column
        )
        delta_row = key[0] - neighbor[0]
        delta_column = key[1] - neighbor[1]
        proposals.append(
            np.asarray(state.points_zyx[neighbor])
            + grow_config.spacing_voxels
            * (delta_row * local_row + delta_column * local_column)
        )
        neighbor_normals.append(normal)
    proposal_array = np.stack(proposals)
    proposal = np.mean(proposal_array, axis=0)
    spread = float(np.max(np.linalg.norm(proposal_array - proposal, axis=1)))
    prior = np.sum(neighbor_normals, axis=0)
    prior /= np.linalg.norm(prior)
    return proposal, prior, spread, len(neighbors)


def evaluate_frontier_quad(
    quad: tuple[int, int],
    state: RaggedState,
    accessor: PublicM7BinaryAccessor,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    ragged_config: RaggedConfig,
    *,
    voxel_um: float,
) -> QuadTrial:
    proposed_quads = frozenset((*state.active_quads, quad))
    topology = topology_audit(proposed_quads, seed=state.logical_seed)
    if not topology["pass"]:
        return QuadTrial(quad, False, "topology_gate_failed", (), None, topology, None)
    proposed_vertices = vertices_for_quads(proposed_quads)
    new_keys = sorted(proposed_vertices - set(state.points_zyx))
    if not new_keys:
        return QuadTrial(quad, False, "no_new_vertex", (), None, topology, None)
    normals = _normal_map(state)
    reference_row, reference_column = _reference_basis(state)
    new_points: dict[tuple[int, int], tuple[float, float, float]] = {}
    records: list[Mapping[str, Any]] = []
    # All proposals use the same cycle-start manifold; staged peers never
    # influence one another, making the quad addition genuinely atomic.
    for key in new_keys:
        proposal, prior, spread, neighbor_count = _propose_new_vertex(
            key, state, normals, reference_row, reference_column, grow_config
        )
        record: dict[str, Any] = {
            "logical_vertex": list(key),
            "proposal_zyx": proposal.tolist(),
            "proposal_spread": spread,
            "inward_neighbor_count": neighbor_count,
        }
        if spread > grow_config.maximum_proposal_spread:
            record.update(pass_=False, reason="inward_proposals_incoherent")
            return QuadTrial(
                quad, False, "inward_proposals_incoherent", tuple((*records, record)),
                None, topology, None,
            )
        projection = project_target_to_m7(
            accessor, proposal, prior, grow_config, inward_neighbor_count=neighbor_count
        )
        record.update(
            {
                "projection_pass": projection.passed,
                "projection_reason": projection.reason,
                "foreground_candidate_count": projection.foreground_candidate_count,
                "evaluated_candidate_count": projection.evaluated_candidate_count,
                "passing_candidate_count": projection.passing_candidate_count,
                "ambiguity": projection.ambiguity,
            }
        )
        if not projection.passed or projection.node is None:
            record["pass"] = False
            return QuadTrial(
                quad, False, projection.reason, tuple((*records, record)), None, topology, None
            )
        initial = canonical_trilinear_transect(
            accessor,
            projection.node.candidate_voxel_zyx,
            projection.node.normal_zyx,
            grow_config.transect_half_length,
        )
        projected = canonical_trilinear_transect(
            accessor,
            projection.node.point_zyx,
            projection.node.normal_zyx,
            grow_config.transect_half_length,
        )
        record["canonical_initial_transect"] = initial
        record["canonical_projected_transect"] = projected
        if (
            not initial["pass"]
            or abs(float(initial.get("run_center_offset", math.inf)))
            > grow_config.maximum_initial_run_center_offset
            or not projected["pass"]
            or abs(float(projected.get("run_center_offset", math.inf)))
            > grow_config.maximum_projected_run_center_offset
        ):
            record["pass"] = False
            return QuadTrial(
                quad, False, "canonical_new_vertex_transect_failed",
                tuple((*records, record)), None, topology, None,
            )
        point = tuple(float(value) for value in projection.node.point_zyx)
        new_points[key] = point
        record["pass"] = True
        record["point_zyx"] = list(point)
        record["local_normal_zyx"] = list(projection.node.normal_zyx)
        records.append(record)

    combined_points = {**state.points_zyx, **new_points}
    if induced_quads(combined_points) != proposed_quads:
        return QuadTrial(
            quad, False, "unintended_induced_quad", tuple(records), None, topology, None
        )
    evaluation = evaluate_surface(
        combined_points,
        proposed_quads,
        sampler,
        ragged_config,
        voxel_um=voxel_um,
        require_zero_distortion=True,
    )
    if not evaluation.passed:
        reason = "full_surface_exhaustive_gate_failed"
        return QuadTrial(quad, False, reason, tuple(records), None, topology, None)
    next_generation = max(state.generations.values()) + 1
    generations = {**state.generations, **{key: next_generation for key in new_points}}
    new_state = RaggedState(
        points_zyx=combined_points,
        generations=generations,
        active_quads=proposed_quads,
        logical_seed=state.logical_seed,
        evaluation=evaluation,
    )
    return QuadTrial(
        quad=quad,
        passed=True,
        reason="pass",
        new_vertex_records=tuple(records),
        state=new_state,
        topology=topology,
        area_gain_cm2=evaluation.area_cm2 - state.evaluation.area_cm2,
    )


def quad_rank_key(trial: QuadTrial) -> tuple[float, float, float, tuple[int, int]]:
    if not trial.passed or trial.state is None or trial.area_gain_cm2 is None:
        raise ValueError("only passing quad trials can be ranked")
    positive = trial.state.evaluation.native_reports["positive"]
    return (
        -float(trial.area_gain_cm2),
        float(positive["metric_distortion_p95"]),
        float(positive["max_neighbor_distance_um"]),
        trial.quad,
    )


def grow_ragged(
    bootstrap: BootstrapResult,
    accessor: PublicM7BinaryAccessor,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    ragged_config: RaggedConfig,
    *,
    voxel_um: float,
) -> tuple[RaggedState, tuple[Mapping[str, Any], ...], str]:
    state = RaggedState(
        points_zyx=bootstrap.points,
        generations=bootstrap.generations,
        active_quads=bootstrap.active_quads,
        logical_seed=(0, 0),
        evaluation=bootstrap.evaluation,
    )
    cycles: list[Mapping[str, Any]] = []
    stop_reason = "maximum_cycles_reached_fail_closed"
    for cycle in range(1, ragged_config.maximum_cycles + 1):
        if state.evaluation.area_cm2 + 1e-15 >= ragged_config.maximum_area_cm2:
            stop_reason = "maximum_area_reached"
            break
        frontier = frontier_quads(state.active_quads)
        trials = [
            evaluate_frontier_quad(
                quad,
                state,
                accessor,
                sampler,
                grow_config,
                ragged_config,
                voxel_um=voxel_um,
            )
            for quad in frontier
        ]
        passing = sorted((trial for trial in trials if trial.passed), key=quad_rank_key)
        chosen = passing[0] if passing else None
        cycles.append(
            {
                "cycle": cycle,
                "starting_area_cm2": state.evaluation.area_cm2,
                "frontier_quad_count": len(frontier),
                "passing_quad_count": len(passing),
                "chosen_quad": list(chosen.quad) if chosen else None,
                "ranking_rule": [
                    "maximum added physical area",
                    "minimum full-surface metric distortion p95",
                    "minimum full-surface maximum neighbor distance",
                    "lexicographically smallest logical quad",
                ],
                "trials": [
                    {
                        "quad": list(trial.quad),
                        "pass": trial.passed,
                        "reason": trial.reason,
                        "area_gain_cm2": trial.area_gain_cm2,
                        "new_vertex_records": list(trial.new_vertex_records),
                        "topology": trial.topology,
                    }
                    for trial in trials
                ],
            }
        )
        if chosen is None or chosen.state is None:
            stop_reason = f"cycle_{cycle}_no_progress"
            break
        state = chosen.state
    return state, tuple(cycles), stop_reason


def competing_run_clearance(exhaustive: Mapping[str, Any]) -> Mapping[str, Any]:
    clearances: list[int] = []
    for record in exhaustive.get("failure_records", []):
        runs = record.get("runs_inclusive_offsets", [])
        centered = [run for run in runs if int(run[0]) <= 0 <= int(run[1])]
        competitors = [run for run in runs if run not in centered]
        if len(centered) != 1:
            continue
        center_run = centered[0]
        for run in competitors:
            if int(run[1]) < int(center_run[0]):
                clearances.append(int(center_run[0]) - int(run[1]) - 1)
            elif int(run[0]) > int(center_run[1]):
                clearances.append(int(run[0]) - int(center_run[1]) - 1)
    return {
        "competing_run_count": len(clearances),
        "minimum_empty_offset_clearance_voxels": min(clearances) if clearances else None,
        "clearances_voxels": sorted(clearances),
        "interpretation": (
            "diagnostic only: far adjacent wraps are not automatically a sheet switch; "
            "the strict exhaustive one-run gate remains unchanged in v3"
        ),
    }


def write_artifact(
    output: Path,
    state: RaggedState,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    voxel_um: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    temporary.mkdir()
    try:
        grid_zyx, mask, bounds = grid_from_points(state.points_zyx, state.active_quads)
        xyz = grid_zyx[..., ::-1]
        for index, name in enumerate("xyz"):
            tifffile.imwrite(temporary / f"{name}.tif", xyz[..., index].astype(np.float32))
        tifffile.imwrite(temporary / "mask.tif", mask.astype(np.uint8) * 255)
        generations = np.zeros(mask.shape, dtype=np.uint16)
        r0, _, c0, _ = bounds
        for key, generation in state.generations.items():
            if key in state.points_zyx:
                generations[key[0] - r0, key[1] - c0] = int(generation)
        tifffile.imwrite(temporary / "generations.tif", generations)
        active_xyz = xyz[mask]
        meta = {
            "format": "tifxyz",
            "type": "seg",
            "uuid": output.name,
            "source": ALGORITHM_VERSION,
            "target_volume": "PHerc1447",
            "seed": list(DEFAULT_SEED_XYZ),
            "scale": [0.05, 0.05],
            "bbox": [np.min(active_xyz, axis=0).tolist(), np.max(active_xyz, axis=0).tolist()],
            "area_cm2": state.evaluation.area_cm2,
            "voxel_size_um": voxel_um,
            "stored_shape": list(mask.shape),
            "logical_bounds": list(bounds),
            "active_vertex_count": int(mask.sum()),
            "active_quad_count": len(state.active_quads),
            "mask_semantics": "nonzero vertices incident to the induced active quad complex",
        }
        for name, document in (
            ("meta.json", meta),
            ("growth-provenance.json", provenance),
            ("validation.json", validation),
        ):
            (temporary / name).write_text(
                json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(temporary.iterdir())]
        (temporary / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-area-cm2", type=float, default=0.05)
    parser.add_argument("--maximum-cycles", type=int, default=256)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    started = time.monotonic()
    try:
        raise RuntimeError(
            "aborted scientific gate: exact-one-run across +/-15 was falsified "
            "by the published-surface calibration; use "
            "m7_calibrated_ragged_grower.py"
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        config = RaggedConfig(
            maximum_area_cm2=float(args.maximum_area_cm2),
            maximum_cycles=int(args.maximum_cycles),
        )
        config.validate()
        grow_config = GrowConfig()
        source = load_v2_input(Path(args.v2))
        sampler, m7_input = load_public_sampler(
            str(args.m7_root), str(args.m7_array_path), Path(args.blosc)
        )
        accessor = PublicM7BinaryAccessor(sampler)
        bootstrap = enumerate_clean_bootstrap(
            source, sampler, config, voxel_um=float(args.voxel_um)
        )
        state, cycles, stop_reason = grow_ragged(
            bootstrap,
            accessor,
            sampler,
            grow_config,
            config,
            voxel_um=float(args.voxel_um),
        )
        topology = topology_audit(state.active_quads, seed=state.logical_seed)
        reached_target = state.evaluation.area_cm2 + 1e-15 >= config.maximum_area_cm2
        validation_pass = bool(
            topology["pass"]
            and state.evaluation.passed
            and len(state.points_zyx) >= config.minimum_bootstrap_vertices
        )
        validation = {
            "pass": validation_pass,
            "target_area_reached": reached_target,
            "area_cm2": state.evaluation.area_cm2,
            "active_vertex_count": len(state.points_zyx),
            "active_quad_count": len(state.active_quads),
            "topology": topology,
            "native_reports_by_orientation": state.evaluation.native_reports,
            "support": state.evaluation.support,
            "all_active_vertex_transects": state.evaluation.exhaustive,
            "competing_run_clearance": competing_run_clearance(state.evaluation.exhaustive),
            "sampling_complete": state.evaluation.sampling_complete,
            "frames_sha256": state.evaluation.frames_sha256,
            "sample_valid_sha256": state.evaluation.sample_valid_sha256,
            "rng_gate": "none; exhaustive all-active-vertex cleanliness is mandatory",
            "self_intersection_status": "not localized; unknown; never assumed zero",
            "gates": [
                {"name": "minimum_50_active_vertices", "pass": len(state.points_zyx) >= config.minimum_bootstrap_vertices,
                 "value": len(state.points_zyx), "threshold": config.minimum_bootstrap_vertices},
                {"name": "single_connected_disk_topology", "pass": topology["pass"]},
                {"name": "native_geometry_bounds_support_and_exhaustive_m7", "pass": state.evaluation.passed},
            ],
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_strict_bounded_geometry" if validation_pass else "no_go",
            "verdict_scope": "public-m7 geometry only; not an ink/submission surface",
            "stop_reason": stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "paid_compute_cost_usd": 0.0,
            "inputs": {"v2": source.evidence, "m7": m7_input},
            "ragged_config": asdict(config),
            "grow_config_unchanged_vertex_gates": asdict(grow_config),
            "bootstrap": {
                "method": "exhaustive seed-containing rectangular crops with fresh boundary normals",
                "candidate_count": bootstrap.candidate_count,
                "passing_candidate_count": bootstrap.passing_candidate_count,
                "chosen_stored_rows_half_open": list(bootstrap.rows),
                "chosen_stored_columns_half_open": list(bootstrap.columns),
                "chosen_vertex_count": len(bootstrap.points),
                "chosen_area_cm2": bootstrap.evaluation.area_cm2,
                "candidate_summaries": list(bootstrap.candidate_summaries),
            },
            "growth": {
                "cycle_count": len(cycles),
                "cycles": list(cycles),
                "final_active_vertex_count": len(state.points_zyx),
                "final_active_quad_count": len(state.active_quads),
                "final_area_cm2": state.evaluation.area_cm2,
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "Exactly-one run over +/-15 is intentionally retained as a strict gate, though far adjacent wraps can be physical.",
                "Self-intersection is not localized and is never assumed zero.",
                "No raw CT, ink inference, detector, semantic judgment, paid GPU, or RunPod was used.",
            ],
        }
        write_artifact(
            output, state, validation, provenance, voxel_um=float(args.voxel_um)
        )
    except Exception as error:
        error_path = output.with_name(output.name + ".error.json")
        if error_path.exists():
            raise
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
                 "status": "error", "error_type": type(error).__name__, "error": str(error),
                 "elapsed_seconds": time.monotonic() - started},
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"FAIL CLOSED: {type(error).__name__}: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(output), "pass": validation_pass, "target_area_reached": reached_target,
             "area_cm2": state.evaluation.area_cm2, "active_vertices": len(state.points_zyx),
             "active_quads": len(state.active_quads), "stop_reason": stop_reason},
            sort_keys=True,
        )
    )
    return 0 if validation_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
