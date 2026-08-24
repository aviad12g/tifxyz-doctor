#!/usr/bin/env python3
"""Atomically extend a verified m7-locked TIFFXYZ by rectangular edges.

This v2 experiment starts from the immutable v1.4 7x7 artifact.  In each
cycle it evaluates all four possible complete rectangle edges against the
same accepted surface, rejects an entire edge if any vertex or mesh gate
fails, ranks the fully passing edges by precommitted geometric margins, and
accepts exactly one.  It stops after the first cycle with no passing edge or
when the materialized area reaches 0.05 cm2.  No raw CT or paid compute is
used.
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
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from m7_locked_surface_grower import (
    GrowConfig,
    Node,
    PublicM7BinaryAccessor,
    _array_sha256,
    _json_safe,
    hard_geometry_pass,
    load_public_sampler,
    project_target_to_m7,
    transported_basis_zyx,
    triangulated_area_cm2,
)
from native_surface_sampler import (
    SurfaceGrid,
    estimate_surface_normals,
    validate_surface_geometry,
)
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOXEL_UM,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from select_m7_seed_chunk import contiguous_true_runs
from validate_public_m7_patch import (
    DEFAULT_M7_ROOT,
    PublicZarrChunkSampler,
    evaluate_all_vertex_transects,
    evaluate_valid_vertex_transects,
    support_summary,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-locked-directional-edges-v2.0"
BASE_ALGORITHM_VERSION = "m7-locked-complete-rings-v1.4"
DEFAULT_BASE = Path(
    "outputs/first-letters-geometry/m7-locked-grower/"
    "seed-x3525-y4191-z14071/prototype-v1.4-radius6"
)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/m7-locked-grower/"
    "seed-x3525-y4191-z14071/prototype-v2-directional-0p05cm2"
)
DEFAULT_SEED_XYZ = (3525.0, 4191.0, 14071.0)
DIRECTION_ORDER = ("north", "south", "west", "east")


@dataclass(frozen=True)
class DirectionalConfig:
    maximum_area_cm2: float = 0.05
    minimum_vertex_count: int = 50
    maximum_cycles: int = 64
    rng_seed: int = 1447

    def validate(self) -> None:
        if self.maximum_area_cm2 <= 0:
            raise ValueError("maximum_area_cm2 must be positive")
        if self.minimum_vertex_count < 1:
            raise ValueError("minimum_vertex_count must be positive")
        if self.maximum_cycles < 1:
            raise ValueError("maximum_cycles must be positive")


@dataclass(frozen=True)
class BaseState:
    nodes: Mapping[tuple[int, int], Node]
    generations: Mapping[tuple[int, int], int]
    bounds: tuple[int, int, int, int]
    reference_row_zyx: tuple[float, float, float]
    reference_column_zyx: tuple[float, float, float]
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class DirectionTrial:
    direction: str
    passed: bool
    reason: str
    cells: tuple[Mapping[str, Any], ...]
    trial_nodes: Mapping[tuple[int, int], Node]
    new_bounds: tuple[int, int, int, int]
    area_cm2: float | None
    neighbor_report: Mapping[str, Any] | None
    native_reports: Mapping[str, Any] | None
    rank_metrics: Mapping[str, float] | None


@dataclass(frozen=True)
class DirectionalResult:
    nodes: Mapping[tuple[int, int], Node]
    generations: Mapping[tuple[int, int], int]
    bounds: tuple[int, int, int, int]
    points_zyx: NDArray[np.float64]
    generation_grid: NDArray[np.uint16]
    cycle_records: tuple[Mapping[str, Any], ...]
    stop_reason: str


def canonical_trilinear_transect(
    accessor: Any,
    point_global_zyx: Sequence[float],
    normal_zyx: Sequence[float],
    half_length: int,
) -> Mapping[str, Any]:
    """Apply the canonical uint8-render-equivalent m7 rule to one transect.

    The underlying prediction is binary uint8 (0/255).  We trilinearly
    interpolate it, round/clip back to uint8 exactly as the renderer does,
    then threshold at >0.  This intentionally differs from v1.4's nearest
    Boolean development trace.
    """

    point = np.asarray(point_global_zyx, dtype=np.float64)
    normal = np.asarray(normal_zyx, dtype=np.float64)
    magnitude = float(np.linalg.norm(normal))
    if point.shape != (3,) or normal.shape != (3,) or magnitude <= 1e-12:
        return {"pass": False, "reason": "invalid_point_or_normal", "runs": []}
    normal /= magnitude
    offsets = np.arange(-half_length, half_length + 1, dtype=np.float64)
    coordinates = point[None, :] + offsets[:, None] * normal[None, :]
    lower_global = np.floor(coordinates).astype(np.int64)
    upper_global = np.minimum(
        lower_global + 1, np.asarray(accessor.shape_zyx, dtype=np.int64) - 1
    )
    if np.any(lower_global < 0) or np.any(upper_global >= np.asarray(accessor.shape_zyx)):
        return {"pass": False, "reason": "transect_out_of_bounds", "runs": []}
    roi_lower = np.min(lower_global, axis=0)
    roi_upper = np.max(upper_global, axis=0) + 1
    foreground = accessor.read_roi(roi_lower, roi_upper).astype(np.float64) * 255.0
    lower = lower_global - roi_lower
    upper = upper_global - roi_lower
    fraction = coordinates - lower_global
    values = np.zeros(offsets.shape, dtype=np.float64)
    for zbit in (0, 1):
        wz = fraction[:, 0] if zbit else 1.0 - fraction[:, 0]
        zi = upper[:, 0] if zbit else lower[:, 0]
        for ybit in (0, 1):
            wy = fraction[:, 1] if ybit else 1.0 - fraction[:, 1]
            yi = upper[:, 1] if ybit else lower[:, 1]
            for xbit in (0, 1):
                wx = fraction[:, 2] if xbit else 1.0 - fraction[:, 2]
                xi = upper[:, 2] if xbit else lower[:, 2]
                values += wz * wy * wx * foreground[zi, yi, xi]
    trace = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8) > 0
    runs_half_open = contiguous_true_runs(trace)
    runs = [
        [int(start - half_length), int(stop - 1 - half_length)]
        for start, stop in runs_half_open
    ]
    center = half_length
    centered = [run for run in runs_half_open if run[0] <= center < run[1]]
    passed = len(runs_half_open) == 1 and len(centered) == 1
    result = {
        "pass": passed,
        "reason": "pass" if passed else (
            "competing_foreground_run" if centered and len(runs_half_open) > 1 else "no_center_run"
        ),
        "runs": runs,
        "quantized_trace_sha256": hashlib.sha256(trace.astype(np.uint8).tobytes()).hexdigest(),
    }
    if passed:
        start, stop = runs_half_open[0]
        result["run_center_offset"] = float(
            ((start - half_length) + (stop - 1 - half_length)) / 2.0
        )
    return result


def _node_from_json(document: Mapping[str, Any]) -> Node:
    values = dict(document)
    for key in (
        "point_zyx",
        "normal_zyx",
        "proposal_zyx",
        "candidate_voxel_zyx",
        "initial_run_offsets",
        "projected_run_offsets",
    ):
        values[key] = tuple(values[key])
    return Node(**values)


def _read_manifest(directory: Path) -> Mapping[str, str]:
    manifest = directory / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError(f"missing base manifest: {manifest}")
    hashes: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"base manifest mismatch: {name}")
        hashes[name] = digest
    return hashes


def load_verified_base(directory: Path) -> BaseState:
    """Load v1.4 nodes only after reproducing its immutable evidence hashes."""

    directory = directory.resolve()
    hashes = _read_manifest(directory)
    provenance = json.loads((directory / "growth-provenance.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    meta = json.loads((directory / "meta.json").read_text())
    if provenance.get("algorithm_version") != BASE_ALGORITHM_VERSION:
        raise ValueError("base algorithm version is not the pinned v1.4 artifact")
    if provenance.get("status") != "complete":
        raise ValueError("base growth is incomplete")
    reports = validation.get("native_reports_by_orientation", {})
    if set(reports) != {"positive", "negative"} or not all(
        hard_geometry_pass(report, require_zero_distortion=False)
        for report in reports.values()
    ):
        raise ValueError("base native geometry evidence does not pass")
    if not validation.get("sampling_complete"):
        raise ValueError("base m7 sampling is incomplete")
    if float(validation.get("support", {}).get("support_fraction", -1)) < 0.90:
        raise ValueError("base centered m7 support is below 90%")

    growth = provenance["growth"]
    radius = int(growth["accepted_radius"])
    bounds = (-radius, radius, -radius, radius)
    nodes: dict[tuple[int, int], Node] = {
        (0, 0): _node_from_json(growth["seed"]["projection"]["node"])
    }
    for ring in growth["rings"]:
        if not ring.get("pass"):
            continue
        for record in ring["cells"]:
            nodes[tuple(int(value) for value in record["cell"])] = _node_from_json(
                record["node"]
            )
    expected_count = (2 * radius + 1) ** 2
    if len(nodes) != expected_count:
        raise ValueError(f"base node map is incomplete: {len(nodes)} != {expected_count}")

    x = np.asarray(tifffile.imread(directory / "x.tif"), dtype=np.float64)
    y = np.asarray(tifffile.imread(directory / "y.tif"), dtype=np.float64)
    z = np.asarray(tifffile.imread(directory / "z.tif"), dtype=np.float64)
    generation_grid = np.asarray(tifffile.imread(directory / "generations.tif"))
    if x.shape != (2 * radius + 1, 2 * radius + 1) or y.shape != x.shape or z.shape != x.shape:
        raise ValueError("base TIFFXYZ shape is inconsistent with accepted radius")
    stored_xyz = np.stack((x, y, z), axis=-1)
    generations: dict[tuple[int, int], int] = {}
    for row in range(-radius, radius + 1):
        for column in range(-radius, radius + 1):
            node = nodes[(row, column)]
            saved = stored_xyz[row + radius, column + radius]
            if not np.allclose(saved, np.asarray(node.point_zyx)[::-1], atol=5e-4, rtol=0):
                raise ValueError(f"base TIFFXYZ/provenance mismatch at {(row, column)}")
            generations[(row, column)] = int(generation_grid[row + radius, column + radius])

    seed = growth["seed"]
    return BaseState(
        nodes=nodes,
        generations=generations,
        bounds=bounds,
        reference_row_zyx=tuple(seed["reference_row_zyx"]),
        reference_column_zyx=tuple(seed["reference_column_zyx"]),
        evidence={
            "directory": str(directory),
            "manifest_sha256": sha256_file(directory / "MANIFEST.sha256"),
            "member_sha256": hashes,
            "meta_area_cm2": float(meta["area_cm2"]),
            "validation_sha256": sha256_file(directory / "validation.json"),
            "growth_provenance_sha256": sha256_file(directory / "growth-provenance.json"),
            "loaded_node_count": len(nodes),
            "loaded_bounds": list(bounds),
        },
    )


def compact_rectangle(
    nodes: Mapping[tuple[int, int], Node],
    generations: Mapping[tuple[int, int], int],
    bounds: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.uint16]]:
    r0, r1, c0, c1 = (int(value) for value in bounds)
    points = np.empty((r1 - r0 + 1, c1 - c0 + 1, 3), dtype=np.float64)
    output_generations = np.empty(points.shape[:2], dtype=np.uint16)
    for row in range(r0, r1 + 1):
        for column in range(c0, c1 + 1):
            key = (row, column)
            if key not in nodes or key not in generations:
                raise ValueError(f"rectangle is not complete at {key}")
            points[row - r0, column - c0] = nodes[key].point_zyx
            output_generations[row - r0, column - c0] = generations[key]
    return points, output_generations


def edge_cells(
    bounds: Sequence[int], direction: str
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int, int, int]]:
    r0, r1, c0, c1 = (int(value) for value in bounds)
    if direction == "north":
        return tuple((r0 - 1, c) for c in range(c0, c1 + 1)), (r0 - 1, r1, c0, c1)
    if direction == "south":
        return tuple((r1 + 1, c) for c in range(c0, c1 + 1)), (r0, r1 + 1, c0, c1)
    if direction == "west":
        return tuple((r, c0 - 1) for r in range(r0, r1 + 1)), (r0, r1, c0 - 1, c1)
    if direction == "east":
        return tuple((r, c1 + 1) for r in range(r0, r1 + 1)), (r0, r1, c0, c1 + 1)
    raise ValueError(f"unknown direction: {direction}")


def rectangle_neighbor_checks(
    nodes: Mapping[tuple[int, int], Node], config: GrowConfig
) -> tuple[bool, Mapping[str, Any]]:
    failures: list[dict[str, Any]] = []
    minimum_dot = 1.0
    maximum_distance_error = 0.0
    maximum_normal_fraction = 0.0
    for key in sorted(nodes):
        point = np.asarray(nodes[key].point_zyx)
        normal = np.asarray(nodes[key].normal_zyx)
        for delta in ((0, 1), (1, 0)):
            other_key = (key[0] + delta[0], key[1] + delta[1])
            if other_key not in nodes:
                continue
            other_point = np.asarray(nodes[other_key].point_zyx)
            other_normal = np.asarray(nodes[other_key].normal_zyx)
            dot = abs(float(normal @ other_normal))
            displacement = other_point - point
            distance = float(np.linalg.norm(displacement))
            relative_error = abs(distance - config.spacing_voxels) / config.spacing_voxels
            average_normal = normal + other_normal
            average_normal /= np.linalg.norm(average_normal)
            normal_fraction = abs(float(displacement @ average_normal)) / max(distance, 1e-12)
            minimum_dot = min(minimum_dot, dot)
            maximum_distance_error = max(maximum_distance_error, relative_error)
            maximum_normal_fraction = max(maximum_normal_fraction, normal_fraction)
            if (
                dot + 1e-12 < config.minimum_neighbor_normal_dot
                or relative_error > config.neighbor_distance_relative_tolerance
                or normal_fraction > config.maximum_neighbor_normal_displacement_fraction
            ):
                failures.append(
                    {
                        "first": list(key),
                        "second": list(other_key),
                        "normal_abs_dot": dot,
                        "distance_relative_error": relative_error,
                        "normal_displacement_fraction": normal_fraction,
                    }
                )
    return not failures, {
        "minimum_neighbor_normal_abs_dot": minimum_dot,
        "maximum_neighbor_distance_relative_error": maximum_distance_error,
        "maximum_neighbor_normal_displacement_fraction": maximum_normal_fraction,
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def native_reports(
    points_zyx: NDArray[np.float64],
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
) -> Mapping[str, Any]:
    xyz = points_zyx[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    offsets_um = [value * voxel_um for value in DEFAULT_TRANSECT_OFFSETS]
    return {
        label: validate_surface_geometry(
            surface,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }


def _proposal_from_inward(
    cell: tuple[int, int],
    nodes: Mapping[tuple[int, int], Node],
    reference_row_zyx: Sequence[float],
    reference_column_zyx: Sequence[float],
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
    proposals: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for (neighbor_row, neighbor_column), node in inward:
        local_row, local_column = transported_basis_zyx(
            node.normal_zyx, reference_row_zyx, reference_column_zyx
        )
        proposals.append(
            np.asarray(node.point_zyx)
            + config.spacing_voxels
            * ((row - neighbor_row) * local_row + (column - neighbor_column) * local_column)
        )
        normals.append(np.asarray(node.normal_zyx))
    proposal_array = np.stack(proposals)
    proposal = proposal_array.mean(axis=0)
    spread = float(np.max(np.linalg.norm(proposal_array - proposal, axis=1)))
    prior = np.sum(normals, axis=0)
    prior /= np.linalg.norm(prior)
    return proposal, prior, spread, len(inward)


def evaluate_direction(
    accessor: Any,
    nodes: Mapping[tuple[int, int], Node],
    generations: Mapping[tuple[int, int], int],
    bounds: Sequence[int],
    direction: str,
    reference_row_zyx: Sequence[float],
    reference_column_zyx: Sequence[float],
    config: GrowConfig,
    *,
    voxel_um: float,
) -> DirectionTrial:
    cells, new_bounds = edge_cells(bounds, direction)
    trial_nodes: dict[tuple[int, int], Node] = {}
    cell_records: list[Mapping[str, Any]] = []
    failure_reason = "pass"
    for cell in cells:
        proposal, prior, spread, inward_count = _proposal_from_inward(
            cell, nodes, reference_row_zyx, reference_column_zyx, config
        )
        if spread > config.maximum_proposal_spread:
            failure_reason = "inward_proposals_incoherent"
            cell_records.append(
                {"cell": list(cell), "pass": False, "reason": failure_reason, "proposal_spread": spread}
            )
            break
        projection = project_target_to_m7(
            accessor, proposal, prior, config, inward_neighbor_count=inward_count
        )
        record: dict[str, Any] = {
            "cell": list(cell),
            "pass": projection.passed,
            "reason": projection.reason,
            "proposal_spread": spread,
            "foreground_candidate_count": projection.foreground_candidate_count,
            "evaluated_candidate_count": projection.evaluated_candidate_count,
            "passing_candidate_count": projection.passing_candidate_count,
            "ambiguity": projection.ambiguity,
        }
        if not projection.passed or projection.node is None:
            failure_reason = projection.reason
            cell_records.append(record)
            break
        canonical_initial = canonical_trilinear_transect(
            accessor,
            projection.node.candidate_voxel_zyx,
            projection.node.normal_zyx,
            config.transect_half_length,
        )
        record["canonical_initial_transect"] = canonical_initial
        if (
            not canonical_initial["pass"]
            or abs(float(canonical_initial.get("run_center_offset", math.inf)))
            > config.maximum_initial_run_center_offset
        ):
            suffix = (
                str(canonical_initial["reason"])
                if not canonical_initial["pass"]
                else "center_offset_exceeds_gate"
            )
            failure_reason = "canonical_initial_transect_" + suffix
            record["pass"] = False
            record["reason"] = failure_reason
            cell_records.append(record)
            break
        canonical_trace = canonical_trilinear_transect(
            accessor,
            projection.node.point_zyx,
            projection.node.normal_zyx,
            config.transect_half_length,
        )
        record["canonical_projected_transect"] = canonical_trace
        if (
            not canonical_trace["pass"]
            or abs(float(canonical_trace.get("run_center_offset", math.inf)))
            > config.maximum_projected_run_center_offset
        ):
            suffix = (
                str(canonical_trace["reason"])
                if not canonical_trace["pass"]
                else "center_offset_exceeds_gate"
            )
            failure_reason = "canonical_projected_transect_" + suffix
            record["pass"] = False
            record["reason"] = failure_reason
            cell_records.append(record)
            break
        trial_nodes[cell] = projection.node
        record["node"] = asdict(projection.node)
        cell_records.append(record)

    if len(trial_nodes) != len(cells):
        return DirectionTrial(
            direction, False, failure_reason, tuple(cell_records), {}, new_bounds,
            None, None, None, None
        )
    combined = {**nodes, **trial_nodes}
    next_generation = max(generations.values()) + 1
    combined_generations = {**generations, **{key: next_generation for key in trial_nodes}}
    neighbor_pass, neighbor_report = rectangle_neighbor_checks(combined, config)
    points, _ = compact_rectangle(combined, combined_generations, new_bounds)
    reports = native_reports(
        points, volume_shape_zyx=accessor.shape_zyx, voxel_um=voxel_um
    )
    geometry_pass = all(
        hard_geometry_pass(report, require_zero_distortion=True)
        for report in reports.values()
    )
    if not neighbor_pass:
        return DirectionTrial(
            direction, False, "edge_neighbor_smoothness_failed", tuple(cell_records), {},
            new_bounds, None, neighbor_report, reports, None
        )
    if not geometry_pass:
        return DirectionTrial(
            direction, False, "edge_native_geometry_failed", tuple(cell_records), {},
            new_bounds, None, neighbor_report, reports, None
        )
    area = triangulated_area_cm2(points, voxel_um)
    proposal_distances = [float(record["node"]["proposal_distance"]) for record in cell_records]
    rank_metrics = {
        "minimum_neighbor_normal_abs_dot": float(neighbor_report["minimum_neighbor_normal_abs_dot"]),
        "maximum_neighbor_distance_relative_error": float(neighbor_report["maximum_neighbor_distance_relative_error"]),
        "maximum_neighbor_normal_displacement_fraction": float(neighbor_report["maximum_neighbor_normal_displacement_fraction"]),
        "mean_new_vertex_proposal_distance": float(np.mean(proposal_distances)),
    }
    return DirectionTrial(
        direction, True, "pass", tuple(cell_records), trial_nodes, new_bounds,
        area, neighbor_report, reports, rank_metrics
    )


def direction_rank_key(trial: DirectionTrial) -> tuple[float, float, float, float, int]:
    if not trial.passed or trial.rank_metrics is None:
        raise ValueError("only passing direction trials can be ranked")
    metrics = trial.rank_metrics
    return (
        -float(metrics["minimum_neighbor_normal_abs_dot"]),
        float(metrics["maximum_neighbor_distance_relative_error"]),
        float(metrics["maximum_neighbor_normal_displacement_fraction"]),
        float(metrics["mean_new_vertex_proposal_distance"]),
        DIRECTION_ORDER.index(trial.direction),
    )


def grow_directional_edges(
    accessor: Any,
    base: BaseState,
    grow_config: GrowConfig,
    directional_config: DirectionalConfig,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
) -> DirectionalResult:
    grow_config.validate()
    directional_config.validate()
    nodes = dict(base.nodes)
    generations = dict(base.generations)
    bounds = base.bounds
    records: list[Mapping[str, Any]] = []
    stop_reason = "maximum_cycles_reached"
    for cycle in range(1, directional_config.maximum_cycles + 1):
        trials = [
            evaluate_direction(
                accessor, nodes, generations, bounds, direction,
                base.reference_row_zyx, base.reference_column_zyx, grow_config,
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
                "trials": [asdict(trial) for trial in trials],
                "chosen_direction": chosen.direction if chosen else None,
                "ranking_rule": [
                    "maximum minimum_neighbor_normal_abs_dot",
                    "minimum maximum_neighbor_distance_relative_error",
                    "minimum maximum_neighbor_normal_displacement_fraction",
                    "minimum mean_new_vertex_proposal_distance",
                    "fixed direction order north,south,west,east",
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
        if chosen.area_cm2 is not None and chosen.area_cm2 + 1e-15 >= directional_config.maximum_area_cm2:
            stop_reason = "maximum_area_reached"
            break
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


def final_validation(
    result: DirectionalResult,
    sampler: PublicZarrChunkSampler,
    grow_config: GrowConfig,
    directional_config: DirectionalConfig,
    *,
    voxel_um: float,
) -> Mapping[str, Any]:
    points_xyz = result.points_zyx[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(
        points_xyz[..., 0], points_xyz[..., 1], points_xyz[..., 2]
    )
    normals = estimate_surface_normals(surface, voxel_size_zyx_um=(voxel_um,) * 3)
    reports = native_reports(
        result.points_zyx, volume_shape_zyx=sampler.shape_zyx, voxel_um=voxel_um
    )
    geometry_pass = all(
        hard_geometry_pass(report, require_zero_distortion=False)
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
    deterministic = evaluate_valid_vertex_transects(
        frames, surface.valid, seed=directional_config.rng_seed, sample_count=50
    )
    exhaustive = evaluate_all_vertex_transects(frames, surface.valid)
    sampling_complete = bool(np.all(sample_valid))
    area_cm2 = triangulated_area_cm2(result.points_zyx, voxel_um)
    gates = [
        {"name": "minimum_50_contiguous_rectangle_vertices", "pass": len(result.nodes) >= directional_config.minimum_vertex_count,
         "value": len(result.nodes), "threshold": directional_config.minimum_vertex_count},
        {"name": "native_geometry_both_orientations", "pass": geometry_pass},
        {"name": "all_requested_m7_samples_valid", "pass": sampling_complete},
        {"name": "five_offset_m7_support", "pass": float(support["support_fraction"]) >= grow_config.minimum_final_support_fraction,
         "value": support["support_fraction"], "threshold": grow_config.minimum_final_support_fraction},
        {"name": "deterministic_50_of_50_unique_centered_transects", "pass": bool(deterministic.get("pass", False))},
        {"name": "exhaustive_all_vertex_unique_centered_transects", "pass": bool(exhaustive.get("all_clean", False))},
    ]
    return {
        "pass": all(bool(gate["pass"]) for gate in gates),
        "gates": gates,
        "area_cm2": area_cm2,
        "stored_shape": list(result.points_zyx.shape[:2]),
        "bounds_row_min_max_column_min_max": list(result.bounds),
        "native_reports_by_orientation": reports,
        "support": support,
        "deterministic_50_transects": deterministic,
        "all_vertex_transects": exhaustive,
        "sampling_complete": sampling_complete,
        "quantized_frames_array_sha256": _array_sha256(frames),
        "sample_validity_array_sha256": _array_sha256(sample_valid),
        "normal_orientation_note": "positive/negative are co-equal reversals of one centered stack",
        "self_intersection_status": "not localized; unknown; never assumed zero",
    }


def write_artifact(
    output: Path,
    result: DirectionalResult,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    grow_config: GrowConfig,
    directional_config: DirectionalConfig,
    voxel_um: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
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
            "seed": list(DEFAULT_SEED_XYZ),
            "scale": [1.0 / grow_config.spacing_voxels] * 2,
            "bbox": [np.min(xyz, axis=(0, 1)).tolist(), np.max(xyz, axis=(0, 1)).tolist()],
            "area_cm2": triangulated_area_cm2(result.points_zyx, voxel_um),
            "voxel_size_um": voxel_um,
            "stored_shape": list(result.points_zyx.shape[:2]),
            "logical_bounds": list(result.bounds),
            "grow_config": asdict(grow_config),
            "directional_config": asdict(directional_config),
        }
        documents = {
            "meta.json": meta,
            "growth-provenance.json": provenance,
            "validation.json": validation,
        }
        for name, document in documents.items():
            (temporary / name).write_text(
                json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(temporary.iterdir())
        ]
        (temporary / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-area-cm2", type=float, default=0.05)
    parser.add_argument("--maximum-cycles", type=int, default=64)
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
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        base = load_verified_base(Path(args.base))
        grow_config = GrowConfig()
        directional_config = DirectionalConfig(
            maximum_area_cm2=float(args.maximum_area_cm2),
            maximum_cycles=int(args.maximum_cycles),
        )
        sampler, m7_input = load_public_sampler(
            str(args.m7_root), str(args.m7_array_path), Path(args.blosc)
        )
        accessor = PublicM7BinaryAccessor(sampler)
        result = grow_directional_edges(
            accessor, base, grow_config, directional_config, voxel_um=float(args.voxel_um)
        )
        validation = final_validation(
            result, sampler, grow_config, directional_config, voxel_um=float(args.voxel_um)
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_bounded_directional_prototype" if validation["pass"] else "no_go",
            "verdict_scope": "free-local public-m7 geometry only; not a submission surface",
            "elapsed_seconds": time.monotonic() - started,
            "paid_compute_cost_usd": 0.0,
            "inputs": {"base": base.evidence, "m7": m7_input},
            "grow_config_unchanged_from_v1_4": asdict(grow_config),
            "directional_config": asdict(directional_config),
            "growth": {
                "initial_bounds": list(base.bounds),
                "final_bounds": list(result.bounds),
                "final_shape": list(result.points_zyx.shape[:2]),
                "final_vertex_count": len(result.nodes),
                "stop_reason": result.stop_reason,
                "cycles": list(result.cycle_records),
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "PCA normals and nearest-voxel run projection remain local m7 heuristics.",
                "Self-intersection is not localized and is never assumed zero.",
                "Every direction is evaluated against the same cycle-start rectangle; only one edge is accepted.",
                "No raw CT, ink inference, detector, semantic judgment, paid GPU, or RunPod was used.",
            ],
        }
        write_artifact(
            output, result, validation, provenance,
            grow_config=grow_config,
            directional_config=directional_config,
            voxel_um=float(args.voxel_um),
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
    print(json.dumps({"output": str(output), "pass": validation["pass"], "stop_reason": result.stop_reason,
                      "shape": list(result.points_zyx.shape[:2]), "area_cm2": validation["area_cm2"]}, sort_keys=True))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
