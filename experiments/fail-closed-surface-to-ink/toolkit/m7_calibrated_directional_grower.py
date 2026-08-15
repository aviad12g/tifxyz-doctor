#!/usr/bin/env python3
"""Calibration-corrected directional m7 surface growth.

The old exactly-one-run-across-+/-15 gate was falsified on a clean published
surface (399/400 vertices have multiple runs).  This grower instead selects
the run crossing the proposed foreground point, recenters that run, preserves
all competing runs as diagnostics, and gates identity with empirically
calibrated centered support: >=0.9875 within +/-2 and >=0.8625 at offset 0.
Geometry, bounds, local normal/spacing coherence, and projection-cluster
ambiguity remain fail-closed.  It continues either the immutable v2 rectangle
or a hash-verified calibrated complete-ring checkpoint atomically by complete
edges toward 0.5 cm2, with no raw CT or paid compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from m7_locked_directional_grower import (
    DIRECTION_ORDER,
    DirectionTrial,
    DirectionalConfig,
    DirectionalResult,
    compact_rectangle,
    direction_rank_key,
    edge_cells,
    native_reports,
    rectangle_neighbor_checks,
)
from m7_locked_surface_grower import (
    GrowConfig,
    Node,
    ProjectionResult,
    PublicM7BinaryAccessor,
    _array_sha256,
    _foreground_candidates,
    _json_safe,
    _nearest_integer,
    hard_geometry_pass,
    load_public_sampler,
    projections_are_distinct_sheets,
    transported_basis_zyx,
    triangulated_area_cm2,
)
from native_surface_sampler import SurfaceGrid, estimate_surface_normals
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOLUME_SHAPE_ZYX,
    DEFAULT_VOXEL_UM,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from select_m7_seed_chunk import contiguous_true_runs, estimate_pca_normal
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    DEFAULT_M7_ROOT,
    PublicZarrChunkSampler,
    evaluate_all_vertex_transects,
    support_summary,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-calibrated-directional-edges-v3.1"
DEFAULT_V2 = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v2-directional-0p05cm2-publicrun"
)
DEFAULT_CALIBRATION = Path(
    "outputs/first-letters-geometry/calibration-m7-probe-20250702235910-20x20/summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/m7-locked-grower/seed-x3525-y4191-z14071/"
    "prototype-v3p1-calibrated-directional-0p5cm2"
)


@dataclass(frozen=True)
class CalibrationGate:
    minimum_support_fraction_pm2: float = 0.9875
    minimum_offset0_fraction: float = 0.8625
    minimum_competitor_empty_gap_voxels: int = 1


@dataclass(frozen=True)
class ContinuedBase:
    nodes: Mapping[tuple[int, int], Node]
    generations: Mapping[tuple[int, int], int]
    bounds: tuple[int, int, int, int]
    reference_row_zyx: tuple[float, float, float]
    reference_column_zyx: tuple[float, float, float]
    evidence: Mapping[str, Any]


def _read_manifest(directory: Path) -> Mapping[str, str]:
    hashes: dict[str, str] = {}
    for line in (directory / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"v2 manifest mismatch: {name}")
        hashes[name] = digest
    return hashes


def _tangent_basis(normal_zyx: Sequence[float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal = np.asarray(normal_zyx, dtype=np.float64)
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


def load_continued_base(directory: Path, *, voxel_um: float) -> ContinuedBase:
    directory = directory.resolve()
    hashes = _read_manifest(directory)
    meta = json.loads((directory / "meta.json").read_text())
    provenance = json.loads((directory / "growth-provenance.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    algorithm = provenance.get("algorithm_version")
    if algorithm == "m7-locked-directional-edges-v2.0":
        required = {
            "minimum_50_contiguous_rectangle_vertices",
            "native_geometry_both_orientations",
            "all_requested_m7_samples_valid",
            "five_offset_m7_support",
        }
        observed = {gate["name"]: bool(gate["pass"]) for gate in validation["gates"]}
        if not required.issubset(observed) or not all(observed[name] for name in required):
            raise ValueError("v2 prerequisite geometry/support gate failed")
        bounds = tuple(int(value) for value in meta["logical_bounds"])
        base_kind = "immutable_directional_v2"
    elif algorithm == "m7-calibrated-complete-rings-v1.0":
        radius = int(validation.get("accepted_radius", -1))
        stored_shape = tuple(int(value) for value in validation.get("stored_shape", ()))
        expected_vertices = (2 * radius + 1) ** 2
        support = validation.get("support", {})
        records = list(validation.get("selected_run_records", ()))
        reports = validation.get("native_reports_by_orientation", {})
        competitor_clear = all(
            record.get("nearest_competitor_empty_gap_voxels") is None
            or int(record["nearest_competitor_empty_gap_voxels"]) >= 1
            for record in records
        )
        checkpoint_pass = bool(
            radius >= 2
            and stored_shape == (2 * radius + 1, 2 * radius + 1)
            and validation.get("geometry_pass") is True
            and validation.get("sampling_complete") is True
            and float(support.get("support_fraction", -1.0)) == 1.0
            and float(support.get("offset0_support_fraction", -1.0)) == 1.0
            and int(validation.get("centered_selected_run_pass_count", -1))
            == expected_vertices
            and len(records) == expected_vertices
            and competitor_clear
            and set(reports) == {"positive", "negative"}
        )
        if not checkpoint_pass:
            raise ValueError("calibrated complete-ring checkpoint gates failed")
        bounds = (-radius, radius, -radius, radius)
        base_kind = "calibrated_complete_ring_checkpoint"
    else:
        raise ValueError("base is not a supported calibrated continuation artifact")
    asset = load_tifxyz_asset(directory, resolution="stored")
    surface = asset.surface
    if not bool(surface.valid.all()):
        raise ValueError("v2 rectangle is unexpectedly masked")
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    if not bool(normals.valid.all()):
        raise ValueError("v2 rectangle has invalid native normals")
    generations_grid = np.asarray(tifffile.imread(directory / "generations.tif"), dtype=np.uint16)
    r0, r1, c0, c1 = bounds
    nodes: dict[tuple[int, int], Node] = {}
    generations: dict[tuple[int, int], int] = {}
    for row in range(r0, r1 + 1):
        for column in range(c0, c1 + 1):
            stored = (row - r0, column - c0)
            point = tuple(float(value) for value in surface.points_xyz[stored][::-1])
            normal = tuple(float(value) for value in normals.positive_xyz[stored][::-1])
            nodes[(row, column)] = Node(
                point_zyx=point,
                normal_zyx=normal,
                proposal_zyx=point,
                candidate_voxel_zyx=tuple(int(value) for value in _nearest_integer(np.asarray(point))),
                initial_run_offsets=(0, 0),
                initial_run_center_offset=0.0,
                projected_run_offsets=(0, 0),
                projected_run_center_offset=0.0,
                proposal_distance=0.0,
                pca_normal_ratio=0.0,
                pca_tangent_ratio=1.0,
                inward_neighbor_count=0,
            )
            generations[(row, column)] = int(generations_grid[stored])
    reference_row, reference_column = _tangent_basis(nodes[(0, 0)].normal_zyx)
    return ContinuedBase(
        nodes=nodes,
        generations=generations,
        bounds=bounds,
        reference_row_zyx=tuple(reference_row),
        reference_column_zyx=tuple(reference_column),
        evidence={
            "base_kind": base_kind,
            "algorithm_version": algorithm,
            "directory": str(directory),
            "manifest_sha256": sha256_file(directory / "MANIFEST.sha256"),
            "member_sha256": hashes,
            "stored_asset_hashes": dict(asset.input_sha256),
            "validation_sha256": sha256_file(directory / "validation.json"),
        },
    )


def _quantized_trace(
    accessor: PublicM7BinaryAccessor,
    point_global_zyx: Sequence[float],
    normal_zyx: Sequence[float],
    half_length: int,
) -> Mapping[str, Any]:
    point = np.asarray(point_global_zyx, dtype=np.float64)
    normal = np.asarray(normal_zyx, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    offsets = np.arange(-half_length, half_length + 1, dtype=np.float64)
    coordinates = point[None, :] + offsets[:, None] * normal[None, :]
    lower_global = np.floor(coordinates).astype(np.int64)
    upper_global = np.minimum(lower_global + 1, np.asarray(accessor.shape_zyx) - 1)
    if np.any(lower_global < 0) or np.any(upper_global >= np.asarray(accessor.shape_zyx)):
        return {"pass": False, "reason": "transect_out_of_bounds", "runs": []}
    roi_lower, roi_upper = np.min(lower_global, axis=0), np.max(upper_global, axis=0) + 1
    foreground = accessor.read_roi(roi_lower, roi_upper).astype(np.float64) * 255.0
    lower, upper = lower_global - roi_lower, upper_global - roi_lower
    fraction = coordinates - lower_global
    values = np.zeros(offsets.size, dtype=np.float64)
    for zbit in (0, 1):
        wz, zi = (fraction[:, 0], upper[:, 0]) if zbit else (1 - fraction[:, 0], lower[:, 0])
        for ybit in (0, 1):
            wy, yi = (fraction[:, 1], upper[:, 1]) if ybit else (1 - fraction[:, 1], lower[:, 1])
            for xbit in (0, 1):
                wx, xi = (fraction[:, 2], upper[:, 2]) if xbit else (1 - fraction[:, 2], lower[:, 2])
                values += wz * wy * wx * foreground[zi, yi, xi]
    trace = np.rint(np.clip(values, 0, 255)).astype(np.uint8) > 0
    raw_runs = contiguous_true_runs(trace)
    runs = [[int(a - half_length), int(b - 1 - half_length)] for a, b in raw_runs]
    selected = [run for run in runs if run[0] <= 0 <= run[1]]
    if len(selected) != 1:
        return {
            "pass": False,
            "reason": "no_unique_run_crossing_offset0",
            "runs": runs,
            "trace_sha256": hashlib.sha256(trace.astype(np.uint8).tobytes()).hexdigest(),
        }
    chosen = selected[0]
    center = 0.5 * (chosen[0] + chosen[1])
    gaps = []
    for run in runs:
        if run == chosen:
            continue
        gaps.append(chosen[0] - run[1] - 1 if run[1] < chosen[0] else run[0] - chosen[1] - 1)
    return {
        "pass": True,
        "reason": "pass",
        "runs": runs,
        "selected_run": chosen,
        "selected_run_center_offset": center,
        "nearest_competitor_empty_gap_voxels": min(gaps) if gaps else None,
        "trace_sha256": hashlib.sha256(trace.astype(np.uint8).tobytes()).hexdigest(),
    }


def project_target_calibrated(
    accessor: PublicM7BinaryAccessor,
    target_global_zyx: Sequence[float],
    prior_normal_zyx: Sequence[float],
    config: GrowConfig,
    gate: CalibrationGate,
    *,
    inward_neighbor_count: int,
) -> ProjectionResult:
    target = np.asarray(target_global_zyx, dtype=np.float64)
    prior = np.asarray(prior_normal_zyx, dtype=np.float64)
    prior /= np.linalg.norm(prior)
    margin = int(
        math.ceil(config.proposal_snap_radius)
        + max(config.pca_radius, config.transect_half_length)
        + 3
    )
    center = _nearest_integer(target)
    lower, upper = center - margin, center + margin + 1
    if np.any(lower < 0) or np.any(upper > np.asarray(accessor.shape_zyx)):
        return ProjectionResult(False, "context_roi_out_of_bounds", None, 0, 0, 0, None)
    foreground = accessor.read_roi(lower, upper)
    candidates, squared = _foreground_candidates(
        foreground, lower, target, config.proposal_snap_radius
    )
    if not candidates.size:
        return ProjectionResult(False, "no_foreground_near_proposal", None, 0, 0, 0, None)
    passing: list[tuple[float, tuple[int, int, int], Node, Mapping[str, Any], Mapping[str, Any]]] = []
    selector = config.selector_config()
    evaluated = min(int(candidates.shape[0]), config.maximum_candidates_evaluated)
    for candidate in candidates[:evaluated]:
        estimate, _ = estimate_pca_normal(foreground, candidate - lower, selector)
        if estimate is None:
            continue
        normal = np.asarray(estimate.normal_zyx, dtype=np.float64)
        if float(normal @ prior) < 0:
            normal = -normal
        if float(normal @ prior) + 1e-12 < config.minimum_neighbor_normal_dot:
            continue
        initial = _quantized_trace(
            accessor, candidate.astype(np.float64), normal, config.transect_half_length
        )
        if not initial["pass"]:
            continue
        initial_center = float(initial["selected_run_center_offset"])
        if abs(initial_center) > config.maximum_initial_run_center_offset:
            continue
        initial_gap = initial["nearest_competitor_empty_gap_voxels"]
        if initial_gap is not None and int(initial_gap) < gate.minimum_competitor_empty_gap_voxels:
            continue
        projected_point = candidate.astype(np.float64) + initial_center * normal
        distance = float(np.linalg.norm(projected_point - target))
        if distance > config.maximum_projection_distance:
            continue
        projected = _quantized_trace(
            accessor, projected_point, normal, config.transect_half_length
        )
        if not projected["pass"]:
            continue
        projected_center = float(projected["selected_run_center_offset"])
        if abs(projected_center) > config.maximum_projected_run_center_offset:
            continue
        projected_gap = projected["nearest_competitor_empty_gap_voxels"]
        if projected_gap is not None and int(projected_gap) < gate.minimum_competitor_empty_gap_voxels:
            continue
        node = Node(
            point_zyx=tuple(float(value) for value in projected_point),
            normal_zyx=tuple(float(value) for value in normal),
            proposal_zyx=tuple(float(value) for value in target),
            candidate_voxel_zyx=tuple(int(value) for value in candidate),
            initial_run_offsets=tuple(int(value) for value in initial["selected_run"]),
            initial_run_center_offset=initial_center,
            projected_run_offsets=tuple(int(value) for value in projected["selected_run"]),
            projected_run_center_offset=projected_center,
            proposal_distance=distance,
            pca_normal_ratio=float(estimate.normal_to_tangent_variance_ratio),
            pca_tangent_ratio=float(estimate.tangent_variance_ratio),
            inward_neighbor_count=inward_neighbor_count,
        )
        passing.append((distance, tuple(int(v) for v in candidate), node, initial, projected))
    if not passing:
        reason = (
            "candidate_budget_exhausted_without_calibrated_center_lock"
            if candidates.shape[0] > evaluated
            else "no_candidate_has_calibrated_center_lock"
        )
        return ProjectionResult(False, reason, None, int(candidates.shape[0]), evaluated, 0, None)
    passing.sort(key=lambda item: (item[0], item[1]))
    best = passing[0]
    tied = [item for item in passing if item[0] <= best[0] + config.ambiguity_distance_tie + 1e-12]
    separated = [
        item for item in tied
        if projections_are_distinct_sheets(
            best[2].point_zyx,
            item[2].point_zyx,
            prior,
            config.ambiguity_projection_separation,
        )
    ]
    if separated:
        return ProjectionResult(
            False,
            "ambiguous_projection_clusters",
            None,
            int(candidates.shape[0]),
            evaluated,
            len(passing),
            {
                "best_point_zyx": list(best[2].point_zyx),
                "competing_point_zyx": list(separated[0][2].point_zyx),
                "best_distance": best[0],
                "competing_distance": separated[0][0],
            },
        )
    return ProjectionResult(
        True,
        "pass",
        best[2],
        int(candidates.shape[0]),
        evaluated,
        len(passing),
        {
            "initial_all_runs": best[3]["runs"],
            "projected_all_runs": best[4]["runs"],
            "projected_nearest_competitor_empty_gap_voxels": best[4]["nearest_competitor_empty_gap_voxels"],
            "run_count_role": "diagnostic only",
        },
    )


def _proposal_for_edge_cell(
    cell: tuple[int, int],
    nodes: Mapping[tuple[int, int], Node],
    reference_row: Sequence[float],
    reference_column: Sequence[float],
    config: GrowConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, int]:
    row, column = cell
    inward = [
        (key, node)
        for key, node in sorted(nodes.items())
        if max(abs(key[0] - row), abs(key[1] - column)) <= 1
    ]
    if not inward:
        raise ValueError("edge cell has no inward neighbor")
    proposals, normals = [], []
    for (nr, nc), node in inward:
        local_row, local_column = transported_basis_zyx(
            node.normal_zyx, reference_row, reference_column
        )
        proposals.append(
            np.asarray(node.point_zyx)
            + config.spacing_voxels
            * ((row - nr) * local_row + (column - nc) * local_column)
        )
        normals.append(np.asarray(node.normal_zyx))
    array = np.stack(proposals)
    proposal = np.mean(array, axis=0)
    spread = float(np.max(np.linalg.norm(array - proposal, axis=1)))
    prior = np.sum(normals, axis=0)
    prior /= np.linalg.norm(prior)
    return proposal, prior, spread, len(inward)


def surface_calibrated_gate(
    points_zyx: NDArray[np.float64],
    sampler: PublicZarrChunkSampler,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> Mapping[str, Any]:
    xyz = points_zyx[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    reports = native_reports(points_zyx, volume_shape_zyx=sampler.shape_zyx, voxel_um=voxel_um)
    geometry_pass = all(
        hard_geometry_pass(report, require_zero_distortion=True)
        for report in reports.values()
    )
    frames, sample_valid = sample_offsets(
        sampler,
        surface,
        normals.positive_xyz,
        surface.valid & normals.valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    support_indices = [DEFAULT_TRANSECT_OFFSETS.index(value) for value in DEFAULT_SUPPORT_OFFSETS]
    support = support_summary(frames[support_indices], DEFAULT_SUPPORT_OFFSETS, surface.valid)
    runs = evaluate_all_vertex_transects(frames, surface.valid)
    clearances: list[int] = []
    selected_center_offsets: list[float] = []
    selected_run_records: list[Mapping[str, Any]] = []
    no_selected_run = 0
    for row in range(surface.shape[0]):
        for column in range(surface.shape[1]):
            trace = frames[:, row, column] > 0
            raw = contiguous_true_runs(trace)
            intervals = [[a - 15, b - 1 - 15] for a, b in raw]
            selected = [interval for interval in intervals if interval[0] <= 0 <= interval[1]]
            if len(selected) != 1:
                no_selected_run += 1
                selected_run_records.append(
                    {
                        "row": row,
                        "column": column,
                        "sample_valid": bool(np.all(sample_valid[:, row, column])),
                        "pass": False,
                        "reason": "no_unique_run_crossing_offset0",
                        "runs_inclusive_offsets": intervals,
                    }
                )
                continue
            chosen = selected[0]
            selected_center = 0.5 * (chosen[0] + chosen[1])
            selected_center_offsets.append(selected_center)
            gaps = [
                chosen[0] - interval[1] - 1
                if interval[1] < chosen[0]
                else interval[0] - chosen[1] - 1
                for interval in intervals
                if interval != chosen
            ]
            if gaps:
                clearances.append(min(gaps))
            selected_run_records.append(
                {
                    "row": row,
                    "column": column,
                    "sample_valid": bool(np.all(sample_valid[:, row, column])),
                    "pass": True,
                    "reason": "pass",
                    "runs_inclusive_offsets": intervals,
                    "selected_run_inclusive_offsets": chosen,
                    "selected_run_center_offset": selected_center,
                    "nearest_competitor_empty_gap_voxels": min(gaps) if gaps else None,
                }
            )
    passed = bool(
        geometry_pass
        and np.all(sample_valid)
        and float(support["support_fraction"]) >= gate.minimum_support_fraction_pm2
        and float(support["offset0_support_fraction"]) >= gate.minimum_offset0_fraction
    )
    return {
        "pass": passed,
        "geometry_pass": geometry_pass,
        "native_reports_by_orientation": reports,
        "sampling_complete": bool(np.all(sample_valid)),
        "support": support,
        "full_pm15_run_diagnostic": runs,
        "centered_selected_run_diagnostic": {
            "no_unique_offset0_crossing_run_count": no_selected_run,
            "selected_run_center_offset_min": min(selected_center_offsets) if selected_center_offsets else None,
            "selected_run_center_offset_max": max(selected_center_offsets) if selected_center_offsets else None,
            "nearest_competitor_empty_gap_min": min(clearances) if clearances else None,
            "nearest_competitor_empty_gap_median": float(np.median(clearances)) if clearances else None,
            "nearest_competitor_empty_gap_max": max(clearances) if clearances else None,
            "clearance_count": len(clearances),
        },
        "selected_run_records": selected_run_records,
        "frames_sha256": _array_sha256(frames),
        "sample_valid_sha256": _array_sha256(sample_valid),
    }


def evaluate_calibrated_direction(
    direction: str,
    nodes: Mapping[tuple[int, int], Node],
    generations: Mapping[tuple[int, int], int],
    bounds: Sequence[int],
    reference_row: Sequence[float],
    reference_column: Sequence[float],
    accessor: PublicM7BinaryAccessor,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> DirectionTrial:
    cells, new_bounds = edge_cells(bounds, direction)
    trial_nodes: dict[tuple[int, int], Node] = {}
    records: list[Mapping[str, Any]] = []
    reason = "pass"
    for cell in cells:
        proposal, prior, spread, count = _proposal_for_edge_cell(
            cell, nodes, reference_row, reference_column, grow_config
        )
        if spread > grow_config.maximum_proposal_spread:
            reason = "inward_proposals_incoherent"
            records.append(
                {"cell": list(cell), "pass": False, "reason": reason, "proposal_spread": spread}
            )
            break
        projection = project_target_calibrated(
            accessor,
            proposal,
            prior,
            grow_config,
            gate,
            inward_neighbor_count=count,
        )
        record = {
            "cell": list(cell),
            "pass": projection.passed,
            "reason": projection.reason,
            "proposal_spread": spread,
            "foreground_candidate_count": projection.foreground_candidate_count,
            "evaluated_candidate_count": projection.evaluated_candidate_count,
            "passing_candidate_count": projection.passing_candidate_count,
            "diagnostic": projection.ambiguity,
        }
        if not projection.passed or projection.node is None:
            reason = projection.reason
            records.append(record)
            break
        trial_nodes[cell] = projection.node
        record["node"] = asdict(projection.node)
        records.append(record)
    if len(trial_nodes) != len(cells):
        return DirectionTrial(direction, False, reason, tuple(records), {}, new_bounds, None, None, None, None)
    combined = {**nodes, **trial_nodes}
    next_generation = max(generations.values()) + 1
    combined_generations = {**generations, **{key: next_generation for key in trial_nodes}}
    neighbor_pass, neighbor_report = rectangle_neighbor_checks(combined, grow_config)
    points, _ = compact_rectangle(combined, combined_generations, new_bounds)
    if not neighbor_pass:
        return DirectionTrial(
            direction, False, "edge_neighbor_smoothness_failed", tuple(records), {},
            new_bounds, None, neighbor_report, None, None,
        )
    calibrated = surface_calibrated_gate(points, sampler, gate, voxel_um=voxel_um)
    if not calibrated["pass"]:
        return DirectionTrial(
            direction, False, "edge_full_surface_calibrated_gate_failed", tuple(records), {},
            new_bounds, None, neighbor_report, calibrated, None,
        )
    area = triangulated_area_cm2(points, voxel_um)
    distances = [float(record["node"]["proposal_distance"]) for record in records]
    rank_metrics = {
        "minimum_neighbor_normal_abs_dot": float(neighbor_report["minimum_neighbor_normal_abs_dot"]),
        "maximum_neighbor_distance_relative_error": float(neighbor_report["maximum_neighbor_distance_relative_error"]),
        "maximum_neighbor_normal_displacement_fraction": float(neighbor_report["maximum_neighbor_normal_displacement_fraction"]),
        "mean_new_vertex_proposal_distance": float(np.mean(distances)),
    }
    return DirectionTrial(
        direction, True, "pass", tuple(records), trial_nodes, new_bounds,
        area, neighbor_report, calibrated, rank_metrics,
    )


def refresh_native_normals(
    nodes: Mapping[tuple[int, int], Node],
    generations: Mapping[tuple[int, int], int],
    bounds: Sequence[int],
    *,
    voxel_um: float,
) -> Mapping[tuple[int, int], Node]:
    points, _ = compact_rectangle(nodes, generations, bounds)
    xyz = points[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    if not bool(normals.valid.all()):
        raise RuntimeError("accepted rectangle produced invalid native normals")
    r0, r1, c0, c1 = (int(value) for value in bounds)
    refreshed: dict[tuple[int, int], Node] = {}
    for row in range(r0, r1 + 1):
        for column in range(c0, c1 + 1):
            stored = (row - r0, column - c0)
            refreshed[(row, column)] = replace(
                nodes[(row, column)],
                normal_zyx=tuple(float(value) for value in normals.positive_xyz[stored][::-1]),
            )
    return refreshed


def grow_calibrated_edges(
    base: ContinuedBase,
    accessor: PublicM7BinaryAccessor,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    directional_config: DirectionalConfig,
    gate: CalibrationGate,
    *,
    voxel_um: float,
) -> DirectionalResult:
    nodes = dict(base.nodes)
    generations = dict(base.generations)
    bounds = base.bounds
    records: list[Mapping[str, Any]] = []
    stop_reason = "maximum_cycles_reached_fail_closed"
    for cycle in range(1, directional_config.maximum_cycles + 1):
        points, _ = compact_rectangle(nodes, generations, bounds)
        if triangulated_area_cm2(points, voxel_um) + 1e-15 >= directional_config.maximum_area_cm2:
            stop_reason = "maximum_area_reached"
            break
        trials = [
            evaluate_calibrated_direction(
                direction,
                nodes,
                generations,
                bounds,
                base.reference_row_zyx,
                base.reference_column_zyx,
                accessor,
                sampler,
                grow_config,
                gate,
                voxel_um=voxel_um,
            )
            for direction in DIRECTION_ORDER
        ]
        passing = sorted((trial for trial in trials if trial.passed), key=direction_rank_key)
        chosen = passing[0] if passing else None
        records.append(
            {
                "cycle": cycle,
                "starting_bounds": list(bounds),
                "chosen_direction": chosen.direction if chosen else None,
                "passing_directions": [trial.direction for trial in passing],
                "trials": [
                    {
                        "direction": trial.direction,
                        "pass": trial.passed,
                        "reason": trial.reason,
                        "new_bounds": list(trial.new_bounds),
                        "area_cm2": trial.area_cm2,
                        "cells": list(trial.cells),
                        "neighbor_report": trial.neighbor_report,
                        "calibrated_surface_gate": trial.native_reports,
                        "rank_metrics": trial.rank_metrics,
                    }
                    for trial in trials
                ],
            }
        )
        if chosen is None:
            stop_reason = f"cycle_{cycle}_no_expand"
            break
        next_generation = max(generations.values()) + 1
        nodes.update(chosen.trial_nodes)
        generations.update({key: next_generation for key in chosen.trial_nodes})
        bounds = chosen.new_bounds
        nodes = dict(
            refresh_native_normals(nodes, generations, bounds, voxel_um=voxel_um)
        )
    points, generation_grid = compact_rectangle(nodes, generations, bounds)
    return DirectionalResult(
        nodes=nodes,
        generations=generations,
        bounds=bounds,
        points_zyx=points,
        generation_grid=generation_grid,
        cycle_records=tuple(records),
        stop_reason=stop_reason,
    )


def write_artifact(
    output: Path,
    result: DirectionalResult,
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
        xyz = result.points_zyx[..., ::-1]
        for index, name in enumerate("xyz"):
            tifffile.imwrite(temporary / f"{name}.tif", xyz[..., index].astype(np.float32))
        tifffile.imwrite(temporary / "generations.tif", result.generation_grid)
        meta = {
            "format": "tifxyz",
            "type": "seg",
            "uuid": output.name,
            "source": ALGORITHM_VERSION,
            "target_volume": "PHerc1447",
            "seed": [3525.0, 4191.0, 14071.0],
            "scale": [0.05, 0.05],
            "bbox": [np.min(xyz, axis=(0, 1)).tolist(), np.max(xyz, axis=(0, 1)).tolist()],
            "area_cm2": triangulated_area_cm2(result.points_zyx, voxel_um),
            "voxel_size_um": voxel_um,
            "stored_shape": list(result.points_zyx.shape[:2]),
            "logical_bounds": list(result.bounds),
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
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-area-cm2", type=float, default=0.5)
    parser.add_argument("--maximum-cycles", type=int, default=128)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    started = time.monotonic()
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        calibration_path = Path(args.calibration)
        calibration = json.loads(calibration_path.read_text())
        conclusion = calibration["conclusion"]
        gate = CalibrationGate(
            minimum_support_fraction_pm2=float(
                conclusion["calibrated_minimum_plus_minus_2_support_fraction"]
            ),
            minimum_offset0_fraction=float(
                conclusion["calibrated_minimum_offset0_support_fraction"]
            ),
        )
        grow_config = GrowConfig()
        directional_config = DirectionalConfig(
            maximum_area_cm2=float(args.maximum_area_cm2),
            minimum_vertex_count=50,
            maximum_cycles=int(args.maximum_cycles),
        )
        base = load_continued_base(Path(args.v2), voxel_um=float(args.voxel_um))
        sampler, m7_input = load_public_sampler(
            str(args.m7_root),
            str(args.m7_array_path),
            Path(args.blosc),
            args.volume_shape_zyx,
        )
        accessor = PublicM7BinaryAccessor(sampler)
        result = grow_calibrated_edges(
            base,
            accessor,
            sampler,
            grow_config,
            directional_config,
            gate,
            voxel_um=float(args.voxel_um),
        )
        final_gate = surface_calibrated_gate(
            result.points_zyx, sampler, gate, voxel_um=float(args.voxel_um)
        )
        area = triangulated_area_cm2(result.points_zyx, float(args.voxel_um))
        reached_target = area + 1e-15 >= directional_config.maximum_area_cm2
        validation = {
            **final_gate,
            "pass": final_gate["pass"],
            "area_cm2": area,
            "target_area_cm2": directional_config.maximum_area_cm2,
            "target_area_reached": reached_target,
            "stored_shape": list(result.points_zyx.shape[:2]),
            "logical_bounds": list(result.bounds),
            "vertex_count": len(result.nodes),
            "run_count_gate": "none; full +/-15 run count is diagnostic only",
            "identity_gate": {
                "minimum_pm2_support_fraction": gate.minimum_support_fraction_pm2,
                "minimum_offset0_support_fraction": gate.minimum_offset0_fraction,
                "observed_pm2_support_fraction": final_gate["support"]["support_fraction"],
                "observed_offset0_support_fraction": final_gate["support"]["offset0_support_fraction"],
            },
            "self_intersection_status": "not localized; unknown; never assumed zero",
        }
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_calibrated_geometry" if final_gate["pass"] else "no_go",
            "verdict_scope": "public-m7 geometry only; raw fiber QC required before ink detection",
            "stop_reason": result.stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "paid_compute_cost_usd": 0.0,
            "inputs": {
                "v2": base.evidence,
                "calibration_summary": {
                    "path": str(calibration_path.resolve()),
                    "sha256": sha256_file(calibration_path),
                    "source_validation_sha256": calibration["source_validation"]["original_validation_sha256"],
                },
                "m7": m7_input,
            },
            "calibration_gate": asdict(gate),
            "calibration_rationale": {
                "published_control_multi_run_vertices": 399,
                "published_control_vertex_count": 400,
                "published_control_pm2_support_fraction": 0.9875,
                "published_control_offset0_support_fraction": 0.8625,
                "published_control_competitor_gap_quantiles_voxels": [1, 4, 6, 7, 9, 11, 15],
                "v2_competitor_gaps_voxels": [8, 10, 11, 11, 12],
                "conclusion": "multiple far runs are expected adjacent wraps, not a standalone sheet-switch veto",
            },
            "grow_config": asdict(grow_config),
            "directional_config": asdict(directional_config),
            "growth": {
                "initial_bounds": list(base.bounds),
                "final_bounds": list(result.bounds),
                "final_shape": list(result.points_zyx.shape[:2]),
                "final_vertex_count": len(result.nodes),
                "cycle_count": len(result.cycle_records),
                "cycles": list(result.cycle_records),
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "Run multiplicity across +/-15 is retained as a diagnostic and is not an identity veto.",
                "Self-intersection is not localized and is never assumed zero.",
                "No raw CT, ink inference, detector, semantic judgment, paid GPU, or RunPod was used.",
            ],
        }
        write_artifact(
            output, result, validation, provenance, voxel_um=float(args.voxel_um)
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
            {"output": str(output), "pass": final_gate["pass"], "target_area_reached": reached_target,
             "area_cm2": area, "shape": list(result.points_zyx.shape[:2]),
             "stop_reason": result.stop_reason},
            sort_keys=True,
        )
    )
    return 0 if final_gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
