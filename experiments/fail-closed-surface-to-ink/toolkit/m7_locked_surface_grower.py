#!/usr/bin/env python3
"""Conservative free-local PHerc1447 surface growth locked to public binary m7.

This prototype is intentionally different from default GrowPatch.  It never
propagates a point solely from a normal grid.  Starting from one foreground
seed, it estimates a local PCA normal, constructs a deterministic tangent
frame, and grows complete concentric square rings at approximately 20-voxel
spacing.  Every proposed vertex must be re-anchored to exactly one centered
m7 foreground run, agree with all inward neighbours, and preserve a smooth,
unfolded native TIFFXYZ mesh.  Any ambiguity rejects the current ring; no
partial ring is materialized.

The output is a bounded stored-grid TIFFXYZ geometry prototype only.  It uses
no raw CT, ink model, detector, semantic inference, paid GPU, or RunPod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from native_surface_sampler import (
    SurfaceGrid,
    estimate_surface_normals,
    validate_surface_geometry,
)
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOLUME_SHAPE_ZYX,
    DEFAULT_VOXEL_UM,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import (
    ZarrV2ArraySpec,
    fetch_json,
    intersecting_chunk_indices,
    sha256_file,
)
from select_m7_seed_chunk import (
    SelectorConfig,
    contiguous_true_runs,
    estimate_pca_normal,
)
from validate_public_m7_patch import (
    DEFAULT_M7_ROOT,
    PublicZarrChunkSampler,
    evaluate_all_vertex_transects,
    evaluate_valid_vertex_transects,
    support_summary,
    validate_metadata_axes,
)


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "m7-locked-complete-rings-v1.4"
DEFAULT_SEED_XYZ = (3525.0, 4191.0, 14071.0)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/m7-locked-grower/"
    "seed-x3525-y4191-z14071/prototype-v1.4-radius6"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(document), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _array_sha256(array: NDArray[Any]) -> str:
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(source.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(source.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(source.tobytes())
    return digest.hexdigest()


class BinaryAccessor(Protocol):
    shape_zyx: tuple[int, int, int]

    def read_roi(
        self, lower_zyx: Sequence[int], upper_zyx: Sequence[int]
    ) -> NDArray[np.bool_]: ...


class PublicM7BinaryAccessor:
    """Dense Boolean ROI reads backed by PublicZarrChunkSampler's exact cache."""

    def __init__(self, sampler: PublicZarrChunkSampler) -> None:
        self.sampler = sampler
        self.shape_zyx = tuple(int(value) for value in sampler.shape_zyx)
        self.chunks_zyx = tuple(int(value) for value in sampler.chunks_zyx)
        self.roi_read_count = 0

    def read_roi(
        self, lower_zyx: Sequence[int], upper_zyx: Sequence[int]
    ) -> NDArray[np.bool_]:
        lower = np.asarray(lower_zyx, dtype=np.int64)
        upper = np.asarray(upper_zyx, dtype=np.int64)
        shape = np.asarray(self.shape_zyx, dtype=np.int64)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(lower < 0)
            or np.any(upper <= lower)
            or np.any(upper > shape)
        ):
            raise ValueError("m7 ROI is invalid or outside the public volume")
        indices = intersecting_chunk_indices(lower, upper, self.chunks_zyx)
        index_array = np.asarray(indices, dtype=np.int64)
        self.sampler._ensure_chunks(index_array)  # exact cached loader reused intentionally
        output = np.zeros(tuple(int(value) for value in upper - lower), dtype=bool)
        chunks = np.asarray(self.chunks_zyx, dtype=np.int64)
        for index in indices:
            chunk_index = np.asarray(index, dtype=np.int64)
            chunk_start = chunk_index * chunks
            chunk = self.sampler._cache[index]
            chunk_stop = chunk_start + np.asarray(chunk.shape, dtype=np.int64)
            overlap_start = np.maximum(lower, chunk_start)
            overlap_stop = np.minimum(upper, chunk_stop)
            if np.any(overlap_stop <= overlap_start):
                continue
            destination = tuple(
                slice(int(left), int(right))
                for left, right in zip(overlap_start - lower, overlap_stop - lower)
            )
            source = tuple(
                slice(int(left), int(right))
                for left, right in zip(
                    overlap_start - chunk_start, overlap_stop - chunk_start
                )
            )
            output[destination] = np.asarray(chunk[source]) > 0
        self.roi_read_count += 1
        return output


@dataclass(frozen=True)
class GrowConfig:
    target_radius: int = 6
    minimum_output_radius: int = 2
    spacing_voxels: float = 20.0
    proposal_snap_radius: float = 6.0
    pca_radius: int = 5
    pca_min_points: int = 20
    maximum_normal_to_tangent_variance_ratio: float = 0.25
    minimum_tangent_variance_ratio: float = 0.05
    transect_half_length: int = 15
    maximum_initial_run_center_offset: float = 2.0
    maximum_projected_run_center_offset: float = 1.0
    minimum_neighbor_normal_dot: float = 0.97
    maximum_projection_distance: float = 7.0
    maximum_proposal_spread: float = 5.0
    neighbor_distance_relative_tolerance: float = 0.30
    maximum_neighbor_normal_displacement_fraction: float = 0.25
    ambiguity_distance_tie: float = 0.5
    ambiguity_projection_separation: float = 2.5
    maximum_candidates_evaluated: int = 512
    minimum_final_support_fraction: float = 0.90
    rng_seed: int = 1447

    def validate(self) -> None:
        if self.target_radius < 1:
            raise ValueError("target_radius must be positive")
        if not 1 <= self.minimum_output_radius <= self.target_radius:
            raise ValueError("minimum_output_radius must be in [1,target_radius]")
        for name in (
            "spacing_voxels",
            "proposal_snap_radius",
            "maximum_projection_distance",
            "maximum_proposal_spread",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.minimum_neighbor_normal_dot <= 1:
            raise ValueError("minimum_neighbor_normal_dot must be in (0,1]")
        if not 0 <= self.minimum_final_support_fraction <= 1:
            raise ValueError("minimum_final_support_fraction must be in [0,1]")

    def selector_config(self) -> SelectorConfig:
        return SelectorConfig(
            search_radius=int(math.ceil(self.proposal_snap_radius)),
            pca_radius=self.pca_radius,
            pca_min_points=self.pca_min_points,
            maximum_normal_to_tangent_variance_ratio=(
                self.maximum_normal_to_tangent_variance_ratio
            ),
            minimum_tangent_variance_ratio=self.minimum_tangent_variance_ratio,
            transect_half_length=self.transect_half_length,
            maximum_run_center_offset=self.maximum_initial_run_center_offset,
            tangent_probe_distance=7.0,
            tangent_probe_snap_radius=2.5,
            minimum_normal_abs_dot=self.minimum_neighbor_normal_dot,
        )


@dataclass(frozen=True)
class Node:
    point_zyx: tuple[float, float, float]
    normal_zyx: tuple[float, float, float]
    proposal_zyx: tuple[float, float, float]
    candidate_voxel_zyx: tuple[int, int, int]
    initial_run_offsets: tuple[int, int]
    initial_run_center_offset: float
    projected_run_offsets: tuple[int, int]
    projected_run_center_offset: float
    proposal_distance: float
    pca_normal_ratio: float
    pca_tangent_ratio: float
    inward_neighbor_count: int


@dataclass(frozen=True)
class ProjectionResult:
    passed: bool
    reason: str
    node: Node | None
    foreground_candidate_count: int
    evaluated_candidate_count: int
    passing_candidate_count: int
    ambiguity: Mapping[str, Any] | None


@dataclass(frozen=True)
class GrowResult:
    passed_minimum_size: bool
    accepted_radius: int
    target_radius: int
    points_zyx: NDArray[np.float64]
    normals_zyx: NDArray[np.float64]
    generations: NDArray[np.uint16]
    nodes: Mapping[tuple[int, int], Node]
    seed_record: Mapping[str, Any]
    ring_records: tuple[Mapping[str, Any], ...]
    stop_reason: str


def _canonical_unit(vector: Sequence[float]) -> NDArray[np.float64]:
    result = np.asarray(vector, dtype=np.float64)
    magnitude = float(np.linalg.norm(result))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError("vector must be finite and nonzero")
    result /= magnitude
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0:
        result = -result
    return result


def tangent_basis_zyx(normal_zyx: Sequence[float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal = _canonical_unit(normal_zyx)
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal)))]
    column = _canonical_unit(np.cross(normal, reference))
    row = _canonical_unit(np.cross(normal, column))
    return row, column


def transported_basis_zyx(
    normal_zyx: Sequence[float],
    reference_row_zyx: Sequence[float],
    reference_column_zyx: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal = _canonical_unit(normal_zyx)
    column_reference = np.asarray(reference_column_zyx, dtype=np.float64)
    column = column_reference - float(column_reference @ normal) * normal
    if np.linalg.norm(column) <= 1e-8:
        return tangent_basis_zyx(normal)
    column /= np.linalg.norm(column)
    if float(column @ column_reference) < 0:
        column = -column
    row = np.cross(normal, column)
    row /= np.linalg.norm(row)
    row_reference = np.asarray(reference_row_zyx, dtype=np.float64)
    if float(row @ row_reference) < 0:
        row = -row
        column = -column
    return row, column


def _nearest_integer(value: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.floor(value + 0.5 + 1e-12).astype(np.int64)


def float_transect(
    foreground: NDArray[np.bool_],
    point_local_zyx: Sequence[float],
    normal_zyx: Sequence[float],
    half_length: int,
) -> Mapping[str, Any]:
    point = np.asarray(point_local_zyx, dtype=np.float64)
    normal = _canonical_unit(normal_zyx)
    offsets = np.arange(-half_length, half_length + 1, dtype=np.float64)
    coordinates = point[None, :] + offsets[:, None] * normal[None, :]
    sampled = _nearest_integer(coordinates)
    shape = np.asarray(foreground.shape, dtype=np.int64)
    if np.any(sampled < 0) or np.any(sampled >= shape):
        return {"pass": False, "reason": "transect_out_of_bounds", "runs": []}
    trace = foreground[tuple(sampled.T)]
    runs_half_open = contiguous_true_runs(trace)
    runs = [
        [int(start - half_length), int(stop - 1 - half_length)]
        for start, stop in runs_half_open
    ]
    center = half_length
    crossing = [run for run in runs_half_open if run[0] <= center < run[1]]
    if len(runs_half_open) != 1:
        return {
            "pass": False,
            "reason": "competing_foreground_run" if crossing else "no_center_run",
            "runs": runs,
        }
    if len(crossing) != 1:
        return {"pass": False, "reason": "no_center_run", "runs": runs}
    start, stop = crossing[0]
    center_offset = ((start - half_length) + (stop - 1 - half_length)) / 2.0
    return {
        "pass": True,
        "reason": "pass",
        "runs": runs,
        "run_center_offset": float(center_offset),
        "sampled_local_zyx": sampled.astype(int).tolist(),
    }


def _foreground_candidates(
    foreground: NDArray[np.bool_],
    lower_global_zyx: NDArray[np.int64],
    target_global_zyx: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    local = np.argwhere(foreground).astype(np.int64)
    if local.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float64)
    global_points = local + lower_global_zyx
    deltas = global_points.astype(np.float64) - target_global_zyx[None, :]
    squared = np.einsum("ij,ij->i", deltas, deltas)
    within = squared <= radius * radius + 1e-12
    global_points = global_points[within]
    squared = squared[within]
    if not squared.size:
        return global_points, squared
    order = np.lexsort(
        (global_points[:, 2], global_points[:, 1], global_points[:, 0], squared)
    )
    return global_points[order], squared[order]


def projections_are_distinct_sheets(
    first_zyx: Sequence[float],
    second_zyx: Sequence[float],
    prior_normal_zyx: Sequence[float],
    minimum_normal_separation: float,
) -> bool:
    """Distinguish parallel-sheet ambiguity from ordinary tangent displacement.

    Foreground voxels on one continuous sheet can project to different nearby
    tangent locations.  Only separation along the transported sheet normal is
    evidence of competing sheet centers; tangent separation alone is not.
    """

    delta = np.asarray(second_zyx, dtype=np.float64) - np.asarray(
        first_zyx, dtype=np.float64
    )
    normal = _canonical_unit(prior_normal_zyx)
    return abs(float(delta @ normal)) > float(minimum_normal_separation)


def project_target_to_m7(
    accessor: BinaryAccessor,
    target_global_zyx: Sequence[float],
    prior_normal_zyx: Sequence[float],
    config: GrowConfig,
    *,
    inward_neighbor_count: int,
) -> ProjectionResult:
    """Snap one proposal to one centered local foreground run, or reject it."""

    target = np.asarray(target_global_zyx, dtype=np.float64)
    prior = _canonical_unit(prior_normal_zyx)
    margin = int(
        math.ceil(config.proposal_snap_radius)
        + max(config.pca_radius, config.transect_half_length)
        + 3
    )
    center = _nearest_integer(target)
    lower = center - margin
    upper = center + margin + 1
    shape = np.asarray(accessor.shape_zyx, dtype=np.int64)
    if np.any(lower < 0) or np.any(upper > shape):
        return ProjectionResult(False, "context_roi_out_of_bounds", None, 0, 0, 0, None)
    foreground = accessor.read_roi(lower, upper)
    candidates, squared = _foreground_candidates(
        foreground, lower, target, config.proposal_snap_radius
    )
    if not candidates.size:
        return ProjectionResult(False, "no_foreground_near_proposal", None, 0, 0, 0, None)
    selector = config.selector_config()
    passing: list[tuple[float, tuple[int, int, int], Node]] = []
    evaluated = min(int(candidates.shape[0]), config.maximum_candidates_evaluated)
    for global_candidate, candidate_squared in zip(candidates[:evaluated], squared[:evaluated]):
        local_candidate = global_candidate - lower
        estimate, reason = estimate_pca_normal(foreground, local_candidate, selector)
        if estimate is None:
            continue
        normal = np.asarray(estimate.normal_zyx, dtype=np.float64)
        if float(normal @ prior) < 0:
            normal = -normal
        if float(normal @ prior) + 1e-12 < config.minimum_neighbor_normal_dot:
            continue
        initial = float_transect(
            foreground, local_candidate, normal, config.transect_half_length
        )
        if not initial["pass"]:
            continue
        initial_center = float(initial["run_center_offset"])
        if abs(initial_center) > config.maximum_initial_run_center_offset:
            continue
        projected = global_candidate.astype(np.float64) + initial_center * normal
        proposal_distance = float(np.linalg.norm(projected - target))
        if proposal_distance > config.maximum_projection_distance:
            continue
        projected_local = projected - lower
        centered = float_transect(
            foreground, projected_local, normal, config.transect_half_length
        )
        if not centered["pass"]:
            continue
        centered_offset = float(centered["run_center_offset"])
        if abs(centered_offset) > config.maximum_projected_run_center_offset:
            continue
        node = Node(
            point_zyx=tuple(float(value) for value in projected),
            normal_zyx=tuple(float(value) for value in normal),
            proposal_zyx=tuple(float(value) for value in target),
            candidate_voxel_zyx=tuple(int(value) for value in global_candidate),
            initial_run_offsets=tuple(int(value) for value in initial["runs"][0]),
            initial_run_center_offset=initial_center,
            projected_run_offsets=tuple(int(value) for value in centered["runs"][0]),
            projected_run_center_offset=centered_offset,
            proposal_distance=proposal_distance,
            pca_normal_ratio=float(estimate.normal_to_tangent_variance_ratio),
            pca_tangent_ratio=float(estimate.tangent_variance_ratio),
            inward_neighbor_count=int(inward_neighbor_count),
        )
        passing.append((proposal_distance, tuple(int(v) for v in global_candidate), node))

    if not passing:
        reason = (
            "candidate_budget_exhausted_without_unique_centered_projection"
            if candidates.shape[0] > evaluated
            else "no_candidate_has_unique_centered_projection"
        )
        return ProjectionResult(
            False, reason, None, int(candidates.shape[0]), evaluated, 0, None
        )
    passing.sort(key=lambda item: (item[0], item[1]))
    best_distance, _, best_node = passing[0]
    tied = [
        item for item in passing
        if item[0] <= best_distance + config.ambiguity_distance_tie + 1e-12
    ]
    separated = [
        item for item in tied
        if projections_are_distinct_sheets(
            best_node.point_zyx,
            item[2].point_zyx,
            prior,
            config.ambiguity_projection_separation,
        )
    ]
    if separated:
        ambiguity = {
            "best_point_zyx": list(best_node.point_zyx),
            "best_distance": best_distance,
            "competing_point_zyx": list(separated[0][2].point_zyx),
            "competing_distance": separated[0][0],
        }
        return ProjectionResult(
            False,
            "ambiguous_projection_clusters",
            None,
            int(candidates.shape[0]),
            evaluated,
            len(passing),
            ambiguity,
        )
    return ProjectionResult(
        True,
        "pass",
        best_node,
        int(candidates.shape[0]),
        evaluated,
        len(passing),
        None,
    )


def _compact_surface(
    nodes: Mapping[tuple[int, int], Node], radius: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint16]]:
    size = 2 * radius + 1
    points = np.empty((size, size, 3), dtype=np.float64)
    normals = np.empty_like(points)
    generations = np.empty((size, size), dtype=np.uint16)
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            node = nodes[(row_offset, column_offset)]
            row, column = row_offset + radius, column_offset + radius
            points[row, column] = node.point_zyx
            normals[row, column] = node.normal_zyx
            generations[row, column] = max(abs(row_offset), abs(column_offset)) + 1
    return points, normals, generations


def _native_report_for_points(
    points_zyx: NDArray[np.float64],
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
) -> Mapping[str, Any]:
    points_xyz = points_zyx[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(
        points_xyz[..., 0], points_xyz[..., 1], points_xyz[..., 2]
    )
    offsets_um = [value * voxel_um for value in DEFAULT_TRANSECT_OFFSETS]
    return validate_surface_geometry(
        surface,
        volume_shape_zyx=volume_shape_zyx,
        voxel_size_zyx_um=(voxel_um,) * 3,
        normal_offsets_um=offsets_um,
        normal_sign=1,
    ).to_dict()


def hard_geometry_pass(report: Mapping[str, Any], *, require_zero_distortion: bool) -> bool:
    zero_keys = [
        "discontinuity_edge_count",
        "neighbor_wrap_risk_edge_count",
        "degenerate_quad_count",
        "folded_quad_count",
        "abrupt_normal_flip_edge_count",
    ]
    if require_zero_distortion:
        zero_keys.append("distorted_quad_count")
    return bool(
        all(int(report[key]) == 0 for key in zero_keys)
        and float(report["valid_vertex_fraction"]) == 1.0
        and float(report["in_bounds_vertex_fraction"]) == 1.0
        and float(report["normal_valid_vertex_fraction"]) == 1.0
        and float(report["valid_neighbor_edge_fraction"]) == 1.0
        and float(report["offset_sample_in_bounds_fraction"]) == 1.0
        and float(report["distorted_quad_fraction"]) <= 0.05
    )


def _neighbor_checks(
    nodes: Mapping[tuple[int, int], Node], config: GrowConfig, radius: int
) -> tuple[bool, Mapping[str, Any]]:
    failures: list[dict[str, Any]] = []
    minimum_dot = 1.0
    maximum_distance_error = 0.0
    maximum_normal_fraction = 0.0
    for row in range(-radius, radius + 1):
        for column in range(-radius, radius + 1):
            node = nodes[(row, column)]
            point = np.asarray(node.point_zyx)
            normal = np.asarray(node.normal_zyx)
            # The stored grid's mesh-neighbour graph has only row/column
            # edges.  Diagonal pairs are neither validator edges nor direct
            # propagation constraints; checking them here over-constrains
            # ordinary curvature across sqrt(2) times the grid spacing.
            for delta_row, delta_column in ((0, 1), (1, 0)):
                other_key = (row + delta_row, column + delta_column)
                if other_key not in nodes:
                    continue
                other = nodes[other_key]
                other_point = np.asarray(other.point_zyx)
                other_normal = np.asarray(other.normal_zyx)
                dot = abs(float(normal @ other_normal))
                displacement = other_point - point
                distance = float(np.linalg.norm(displacement))
                expected = config.spacing_voxels * math.sqrt(
                    delta_row * delta_row + delta_column * delta_column
                )
                relative_error = abs(distance - expected) / expected
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
                            "first": [row, column],
                            "second": list(other_key),
                            "normal_abs_dot": dot,
                            "distance_voxels": distance,
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


def grow_complete_rings(
    accessor: BinaryAccessor,
    seed_xyz: Sequence[float],
    config: GrowConfig,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    projector: Callable[..., ProjectionResult] = project_target_to_m7,
) -> GrowResult:
    config.validate()
    seed_zyx = np.asarray(seed_xyz, dtype=np.float64)[::-1]
    seed_projection = projector(
        accessor, seed_zyx, (1.0, 0.0, 0.0), config, inward_neighbor_count=0
    )
    # The unoriented first prior above must not constrain the actual seed PCA.
    # Retry using the seed candidate's deterministic PCA direction when needed.
    if not seed_projection.passed:
        margin = int(
            math.ceil(config.proposal_snap_radius)
            + max(config.pca_radius, config.transect_half_length)
            + 3
        )
        center = _nearest_integer(seed_zyx)
        lower, upper = center - margin, center + margin + 1
        foreground = accessor.read_roi(lower, upper)
        local_seed = _nearest_integer(seed_zyx) - lower
        estimate, reason = estimate_pca_normal(
            foreground, local_seed, config.selector_config()
        )
        if estimate is None:
            raise RuntimeError(f"seed PCA failed: {reason}")
        seed_projection = projector(
            accessor,
            seed_zyx,
            estimate.normal_zyx,
            config,
            inward_neighbor_count=0,
        )
    if not seed_projection.passed or seed_projection.node is None:
        raise RuntimeError(f"seed projection failed closed: {seed_projection.reason}")
    seed_node = seed_projection.node
    reference_row, reference_column = tangent_basis_zyx(seed_node.normal_zyx)
    nodes: dict[tuple[int, int], Node] = {(0, 0): seed_node}
    ring_records: list[Mapping[str, Any]] = []
    accepted_radius = 0
    stop_reason = "target_radius_reached"

    for radius in range(1, config.target_radius + 1):
        raw_ring_cells = sorted(
            (row, column)
            for row in range(-radius, radius + 1)
            for column in range(-radius, radius + 1)
            if max(abs(row), abs(column)) == radius
        )
        # Edge cells have one direct inward mesh neighbour and are evaluated
        # first.  Corners are then predicted from both newly projected edge
        # neighbours plus their diagonal inward anchor, avoiding a single
        # sqrt(2)*spacing extrapolation while preserving all-or-none rings.
        ring_cells = sorted(
            raw_ring_cells,
            key=lambda cell: (
                abs(cell[0]) == radius and abs(cell[1]) == radius,
                cell,
            ),
        )
        trial_nodes: dict[tuple[int, int], Node] = {}
        cell_records: list[dict[str, Any]] = []
        failed = False
        failure_reason = None
        for row, column in ring_cells:
            is_corner = abs(row) == radius and abs(column) == radius
            available_nodes = {**nodes, **trial_nodes} if is_corner else nodes
            inward = [
                (key, node)
                for key, node in available_nodes.items()
                if max(abs(key[0] - row), abs(key[1] - column)) <= 1
                and (
                    max(abs(key[0]), abs(key[1])) < radius
                    or (is_corner and key in trial_nodes)
                )
            ]
            if not inward:
                failed = True
                failure_reason = "no_inward_neighbor"
                cell_records.append({"cell": [row, column], "pass": False, "reason": failure_reason})
                break
            proposals = []
            neighbor_normals = []
            for (neighbor_row, neighbor_column), neighbor in inward:
                local_row, local_column = transported_basis_zyx(
                    neighbor.normal_zyx, reference_row, reference_column
                )
                delta_row = row - neighbor_row
                delta_column = column - neighbor_column
                proposal = np.asarray(neighbor.point_zyx) + config.spacing_voxels * (
                    delta_row * local_row + delta_column * local_column
                )
                proposals.append(proposal)
                neighbor_normals.append(np.asarray(neighbor.normal_zyx))
            proposal_array = np.stack(proposals)
            proposal = proposal_array.mean(axis=0)
            spread = float(np.max(np.linalg.norm(proposal_array - proposal, axis=1)))
            if spread > config.maximum_proposal_spread:
                failed = True
                failure_reason = "inward_proposals_incoherent"
                cell_records.append(
                    {"cell": [row, column], "pass": False, "reason": failure_reason, "spread": spread}
                )
                break
            prior_normal = np.sum(neighbor_normals, axis=0)
            prior_normal /= np.linalg.norm(prior_normal)
            projected = projector(
                accessor,
                proposal,
                prior_normal,
                config,
                inward_neighbor_count=len(inward),
            )
            record = {
                "cell": [row, column],
                "pass": projected.passed,
                "reason": projected.reason,
                "proposal_spread": spread,
                "foreground_candidate_count": projected.foreground_candidate_count,
                "evaluated_candidate_count": projected.evaluated_candidate_count,
                "passing_candidate_count": projected.passing_candidate_count,
                "ambiguity": projected.ambiguity,
            }
            if not projected.passed or projected.node is None:
                failed = True
                failure_reason = projected.reason
                cell_records.append(record)
                break
            trial_nodes[(row, column)] = projected.node
            record["node"] = asdict(projected.node)
            cell_records.append(record)

        if not failed:
            combined = {**nodes, **trial_nodes}
            neighbors_pass, neighbor_report = _neighbor_checks(combined, config, radius)
            points, _, _ = _compact_surface(combined, radius)
            geometry = _native_report_for_points(
                points, volume_shape_zyx=accessor.shape_zyx, voxel_um=voxel_um
            )
            geometry_pass = hard_geometry_pass(geometry, require_zero_distortion=True)
            if not neighbors_pass:
                failed, failure_reason = True, "ring_neighbor_smoothness_failed"
            elif not geometry_pass:
                failed, failure_reason = True, "ring_native_geometry_failed"
        else:
            neighbor_report = None
            geometry = None
            geometry_pass = False

        ring_records.append(
            {
                "radius": radius,
                "requested_cell_count": len(ring_cells),
                "evaluated_cell_count": len(cell_records),
                "accepted_cell_count": len(trial_nodes) if not failed else 0,
                "pass": not failed,
                "failure_reason": failure_reason,
                "cells": cell_records,
                "neighbor_checks": neighbor_report,
                "native_geometry": geometry,
                "native_geometry_pass": geometry_pass,
            }
        )
        if failed:
            stop_reason = f"ring_{radius}_rejected:{failure_reason}"
            break
        nodes.update(trial_nodes)
        accepted_radius = radius

    points, normals, generations = _compact_surface(nodes, accepted_radius)
    return GrowResult(
        passed_minimum_size=accepted_radius >= config.minimum_output_radius,
        accepted_radius=accepted_radius,
        target_radius=config.target_radius,
        points_zyx=points,
        normals_zyx=normals,
        generations=generations,
        nodes=nodes,
        seed_record={
            "requested_seed_xyz": [float(value) for value in seed_xyz],
            "requested_seed_zyx": seed_zyx.tolist(),
            "projection": asdict(seed_projection),
            "reference_row_zyx": reference_row.tolist(),
            "reference_column_zyx": reference_column.tolist(),
        },
        ring_records=tuple(ring_records),
        stop_reason=stop_reason,
    )


def triangulated_area_cm2(points_zyx: NDArray[np.float64], voxel_um: float) -> float:
    points = points_zyx[..., ::-1] * voxel_um
    p00, p01 = points[:-1, :-1], points[:-1, 1:]
    p10, p11 = points[1:, :-1], points[1:, 1:]
    first = np.cross(p01 - p00, p10 - p00)
    second = np.cross(p11 - p10, p11 - p01)
    area_um2 = 0.5 * (np.linalg.norm(first, axis=-1) + np.linalg.norm(second, axis=-1))
    return float(area_um2.sum() / 100_000_000.0)


def final_validation(
    result: GrowResult,
    sampler: PublicZarrChunkSampler,
    config: GrowConfig,
    *,
    voxel_um: float,
) -> Mapping[str, Any]:
    points_xyz = result.points_zyx[..., ::-1]
    surface = SurfaceGrid.from_tifxyz(
        points_xyz[..., 0], points_xyz[..., 1], points_xyz[..., 2]
    )
    native_normals = estimate_surface_normals(
        surface, voxel_size_zyx_um=(voxel_um,) * 3
    )
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
        hard_geometry_pass(report, require_zero_distortion=False)
        for report in reports.values()
    )
    all_frames, sample_valid = sample_offsets(
        sampler,
        surface,
        native_normals.positive_xyz,
        surface.valid & native_normals.valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    sampling_complete = bool(np.all(sample_valid))
    support_indices = [DEFAULT_TRANSECT_OFFSETS.index(value) for value in DEFAULT_SUPPORT_OFFSETS]
    support = support_summary(
        all_frames[support_indices], DEFAULT_SUPPORT_OFFSETS, surface.valid
    )
    deterministic = evaluate_valid_vertex_transects(
        all_frames, surface.valid, seed=config.rng_seed, sample_count=50
    )
    exhaustive = evaluate_all_vertex_transects(all_frames, surface.valid)
    gates = [
        {"name": "minimum_complete_radius", "pass": result.passed_minimum_size,
         "value": result.accepted_radius, "threshold": config.minimum_output_radius},
        {"name": "native_geometry_both_orientations", "pass": geometry_pass},
        {"name": "all_requested_m7_samples_valid", "pass": sampling_complete},
        {"name": "five_offset_m7_support", "pass": support["support_fraction"] >= config.minimum_final_support_fraction,
         "value": support["support_fraction"], "threshold": config.minimum_final_support_fraction},
        {"name": "deterministic_50_of_50_unique_centered_transects", "pass": bool(deterministic.get("pass", False))},
    ]
    return {
        "pass": all(bool(gate["pass"]) for gate in gates),
        "gates": gates,
        "native_reports_by_orientation": reports,
        "support": support,
        "deterministic_50_transects": deterministic,
        "all_vertex_transects": exhaustive,
        "sampling_complete": sampling_complete,
        "quantized_frames_array_sha256": _array_sha256(all_frames),
        "sample_validity_array_sha256": _array_sha256(sample_valid),
        "normal_orientation_note": (
            "positive/negative are co-equal reversals of the same centered ±15 stack"
        ),
        "self_intersection_status": "not localized; unknown; never assumed zero",
    }


def _write_artifact(
    output: Path,
    result: GrowResult,
    validation: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    seed_xyz: Sequence[float],
    config: GrowConfig,
    voxel_um: float,
    target_volume: str | None = None,
    algorithm_version: str = ALGORITHM_VERSION,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        points_xyz = result.points_zyx[..., ::-1]
        for component, name in enumerate("xyz"):
            tifffile.imwrite(
                temporary / f"{name}.tif",
                points_xyz[..., component].astype(np.float32),
                photometric="minisblack",
            )
        tifffile.imwrite(
            temporary / "generations.tif",
            result.generations,
            photometric="minisblack",
        )
        area_cm2 = triangulated_area_cm2(result.points_zyx, voxel_um)
        meta = {
            "format": "tifxyz",
            "type": "seg",
            "uuid": output.name,
            "source": str(algorithm_version),
            "target_volume": (
                str(target_volume)
                if target_volume is not None
                else Path(DEFAULT_M7_ROOT).name
            ),
            "seed": [float(value) for value in seed_xyz],
            "scale": [1.0 / config.spacing_voxels, 1.0 / config.spacing_voxels],
            "bbox": [
                np.min(points_xyz, axis=(0, 1)).tolist(),
                np.max(points_xyz, axis=(0, 1)).tolist(),
            ],
            "area_cm2": area_cm2,
            "voxel_size_um": voxel_um,
            "stored_shape": list(result.points_zyx.shape[:2]),
            "accepted_radius": result.accepted_radius,
            "target_radius": result.target_radius,
            "config": asdict(config),
        }
        (temporary / "meta.json").write_text(
            json.dumps(_json_safe(meta), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "growth-provenance.json").write_text(
            json.dumps(_json_safe(provenance), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "validation.json").write_text(
            json.dumps(_json_safe(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes = []
        for path in sorted(temporary.iterdir()):
            if path.name == "MANIFEST.sha256":
                continue
            hashes.append(f"{sha256_file(path)}  {path.name}")
        (temporary / "MANIFEST.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_public_sampler(
    root: str,
    array_path: str,
    blosc: Path,
    expected_shape_zyx: Sequence[int] = DEFAULT_VOLUME_SHAPE_ZYX,
) -> tuple[PublicZarrChunkSampler, Mapping[str, Any]]:
    root = root.rstrip("/")
    array_path = array_path.strip("/")
    group, group_provenance = fetch_json(root + "/.zgroup")
    metadata, array_provenance = fetch_json(root + "/" + array_path + "/.zarray")
    attributes, attributes_provenance = fetch_json(root + "/.zattrs")
    if group.get("zarr_format") != 2:
        raise ValueError("public m7 is not a Zarr-v2 group")
    validate_metadata_axes(attributes, array_path)
    spec = ZarrV2ArraySpec.from_metadata(metadata)
    expected_shape = tuple(int(value) for value in expected_shape_zyx)
    if len(expected_shape) != 3 or any(value <= 0 for value in expected_shape):
        raise ValueError("expected public m7 shape must contain three positive integers")
    if spec.shape_zyx != expected_shape:
        raise ValueError(
            f"unexpected public m7 shape: {spec.shape_zyx}; expected {expected_shape}"
        )
    if spec.chunks_zyx != (192, 192, 192) or spec.dtype != np.dtype(np.uint8):
        raise ValueError("unexpected public m7 chunks or dtype")
    sampler = PublicZarrChunkSampler(
        root_url=root,
        array_path=array_path,
        spec=spec,
        blosc_path=blosc,
    )
    return sampler, {
        "root": root,
        "array_path": array_path,
        "group_metadata": group_provenance,
        "array_metadata": array_provenance,
        "root_attributes": attributes_provenance,
        "array_spec": asdict(spec),
        "codec": {"path": str(blosc.resolve()), "sha256": sha256_file(blosc)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-xyz", type=float, nargs=3, default=DEFAULT_SEED_XYZ)
    parser.add_argument("--target-radius", type=int, default=6)
    parser.add_argument("--minimum-output-radius", type=int, default=2)
    parser.add_argument("--spacing-voxels", type=float, default=20.0)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument(
        "--target-volume",
        default=Path(DEFAULT_M7_ROOT).name,
        help="explicit target identifier recorded in artifact metadata",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    output = Path(args.output)
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
        config = GrowConfig(
            target_radius=int(args.target_radius),
            minimum_output_radius=int(args.minimum_output_radius),
            spacing_voxels=float(args.spacing_voxels),
        )
        sampler, m7_input = load_public_sampler(
            str(args.m7_root),
            str(args.m7_array_path),
            Path(args.blosc),
            args.volume_shape_zyx,
        )
        accessor = PublicM7BinaryAccessor(sampler)
        result = grow_complete_rings(
            accessor, args.seed_xyz, config, voxel_um=float(args.voxel_um)
        )
        validation = final_validation(
            result, sampler, config, voxel_um=float(args.voxel_um)
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "complete",
            "verdict": "pass_bounded_prototype" if validation["pass"] else "no_go",
            "verdict_scope": (
                "bounded public-m7 geometry prototype only; no raw/ink inference; "
                "not a final submission surface"
            ),
            "paid_compute_cost_usd": 0.0,
            "elapsed_seconds": time.monotonic() - started,
            "inputs": {
                "m7": m7_input,
                "seed_xyz": list(args.seed_xyz),
                "expected_volume_shape_zyx": [
                    int(value) for value in args.volume_shape_zyx
                ],
                "target_volume": str(args.target_volume),
            },
            "config": asdict(config),
            "growth": {
                "accepted_radius": result.accepted_radius,
                "target_radius": result.target_radius,
                "stored_shape": list(result.points_zyx.shape[:2]),
                "passed_minimum_size": result.passed_minimum_size,
                "stop_reason": result.stop_reason,
                "seed": result.seed_record,
                "rings": list(result.ring_records),
            },
            "public_m7_sampling": sampler.manifest(),
            "accessor_roi_read_count": accessor.roi_read_count,
            "limitations": [
                "PCA normals and nearest-voxel run projection are local m7 heuristics.",
                "Self-intersection is not localized and is never assumed zero.",
                "Area remains below the final 0.5 cm2 target in this bounded pilot.",
                "No raw CT, ink inference, letter detection, or semantic judgment was run.",
            ],
        }
        _write_artifact(
            output,
            result,
            validation,
            provenance,
            seed_xyz=args.seed_xyz,
            config=config,
            voxel_um=float(args.voxel_um),
            target_volume=str(args.target_volume),
        )
    except Exception as error:
        error_path = output.with_name(output.name + ".error.json")
        error_document = {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "error",
            "verdict": "no_go",
            "error_type": type(error).__name__,
            "error": str(error),
            "output_requested": str(output),
            "paid_compute_cost_usd": 0.0,
        }
        if not error_path.exists():
            _atomic_json(error_path, error_document)
        print(json.dumps(error_document, sort_keys=True), file=sys.stderr)
        return 1
    summary = {
        "output": str(output),
        "accepted_radius": result.accepted_radius,
        "shape": list(result.points_zyx.shape[:2]),
        "validation_pass": validation["pass"],
        "stop_reason": result.stop_reason,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if validation["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
