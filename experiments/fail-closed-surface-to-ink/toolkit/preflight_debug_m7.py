#!/usr/bin/env python3
"""Free, fail-closed stored-grid geometry and m7 preflight for TIFFXYZ surfaces.

The tool deliberately operates on the *stored* TIFFXYZ control grid.  It is a
cheap falsification preflight, not a substitute for the campaign's full-scale
official renderer.  It reuses the first-letters TIFFXYZ/Zarr loaders and native
normal/validation primitives, localizes every local hard geometry defect that
the native validator exposes, applies a one-grid-cell margin, and performs an
exact maximum-support rectangle search.

No paid compute is used.  Both local paths and public HTTPS TIFFXYZ paths are
accepted.  The Zarr sampler groups points by the chunk containing their lower
trilinear corner and makes one minimal, halo-complete ROI read per group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from native_surface_sampler import (
    SurfaceGrid,
    estimate_surface_normals,
    sample_trilinear_zyx,
    validate_surface_geometry,
)
from tifxyz_render_pipeline import (
    OpenedVolume,
    load_tifxyz_asset,
    open_volume_zyx,
    sanitize_source,
)


SCHEMA_VERSION = 1
DEFAULT_M7 = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/"
    "representations/predictions/surfaces/"
    "20250521151220-surface-20260413222639-surface-m7-L0-th0.2.zarr"
)
DEFAULT_VOLUME_SHAPE_ZYX = (24297, 8343, 8343)
DEFAULT_VOXEL_UM = 8.64
DEFAULT_SUPPORT_OFFSETS = (-2, -1, 0, 1, 2)
DEFAULT_TRANSECT_OFFSETS = tuple(range(-15, 16))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_array(array: NDArray[Any]) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _runs(mask: NDArray[np.bool_]) -> Iterable[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return zip(starts.tolist(), stops.tolist())


def dilate_one_cell(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Eight-connected one-cell binary dilation without a SciPy dependency."""

    source = np.asarray(mask, dtype=bool)
    result = source.copy()
    height, width = source.shape
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            source_row0, source_row1 = (
                max(0, -delta_row),
                min(height, height - delta_row),
            )
            source_col0, source_col1 = (
                max(0, -delta_column),
                min(width, width - delta_column),
            )
            result[
                source_row0 + delta_row : source_row1 + delta_row,
                source_col0 + delta_column : source_col1 + delta_column,
            ] |= source[source_row0:source_row1, source_col0:source_col1]
    return result


def _mark_row_edges(target: NDArray[np.bool_], edges: NDArray[np.bool_]) -> None:
    target[:-1, :] |= edges
    target[1:, :] |= edges


def _mark_column_edges(target: NDArray[np.bool_], edges: NDArray[np.bool_]) -> None:
    target[:, :-1] |= edges
    target[:, 1:] |= edges


def _mark_quads(target: NDArray[np.bool_], quads: NDArray[np.bool_]) -> None:
    target[:-1, :-1] |= quads
    target[:-1, 1:] |= quads
    target[1:, :-1] |= quads
    target[1:, 1:] |= quads


@dataclass(frozen=True)
class GeometryMasks:
    forbidden: NDArray[np.bool_]
    base_defect: NDArray[np.bool_]
    sample_valid: NDArray[np.bool_]
    quad_area_cm2: NDArray[np.float64]
    distorted_quad: NDArray[np.bool_]
    normals_xyz: NDArray[np.float64]
    normal_valid: NDArray[np.bool_]
    summary: Mapping[str, Any]


def derive_geometry_masks(
    surface: SurfaceGrid,
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
    maximum_offset_voxels: int = 15,
    continuity_factor: float = 4.0,
    neighbor_wrap_factor: float = 8.0,
    metric_distortion_limit: float = 4.0,
) -> GeometryMasks:
    """Localize native geometry defects and add the required one-cell margin."""

    shape_zyx = tuple(int(value) for value in volume_shape_zyx)
    if len(shape_zyx) != 3:
        raise ValueError("volume_shape_zyx must contain z,y,x")
    z_size, y_size, x_size = shape_zyx
    points = surface.points_xyz
    finite = np.isfinite(points).all(axis=-1)
    strict_nonnegative = finite & (points >= 0.0).all(axis=-1)
    if np.any(surface.valid & ~strict_nonnegative):
        raise ValueError(
            "surface mask includes a nonfinite or negative TIFFXYZ coordinate"
        )
    strict_validity_matches_loader = bool(
        np.array_equal(surface.valid, strict_nonnegative)
    )
    explicitly_masked_coordinate_count = int(
        np.count_nonzero(strict_nonnegative & ~surface.valid)
    )
    valid = surface.valid
    bounds = valid & (
        (surface.x <= x_size - 1)
        & (surface.y <= y_size - 1)
        & (surface.z <= z_size - 1)
    )
    normals = estimate_surface_normals(
        surface, voxel_size_zyx_um=(voxel_um, voxel_um, voxel_um)
    )
    endpoint_minus = points - float(maximum_offset_voxels) * normals.positive_xyz
    endpoint_plus = points + float(maximum_offset_voxels) * normals.positive_xyz

    def inside(coordinates: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (
            np.isfinite(coordinates).all(axis=-1)
            & (coordinates[..., 0] >= 0.0)
            & (coordinates[..., 0] <= x_size - 1)
            & (coordinates[..., 1] >= 0.0)
            & (coordinates[..., 1] <= y_size - 1)
            & (coordinates[..., 2] >= 0.0)
            & (coordinates[..., 2] <= z_size - 1)
        )

    offset_safe = normals.valid & inside(endpoint_minus) & inside(endpoint_plus)
    sample_valid = valid & bounds & normals.valid & offset_safe
    physical = points * float(voxel_um)

    row_edge_valid = valid[:-1, :] & valid[1:, :]
    column_edge_valid = valid[:, :-1] & valid[:, 1:]
    row_distances = np.linalg.norm(physical[1:] - physical[:-1], axis=-1)
    column_distances = np.linalg.norm(physical[:, 1:] - physical[:, :-1], axis=-1)
    neighbor_values = np.concatenate(
        (row_distances[row_edge_valid], column_distances[column_edge_valid])
    )
    positive = neighbor_values[np.isfinite(neighbor_values) & (neighbor_values > 0.0)]
    median_distance = float(np.median(positive)) if positive.size else float("nan")
    continuity_limit = median_distance * continuity_factor
    wrap_limit = median_distance * neighbor_wrap_factor
    discontinuity_row = row_edge_valid & (row_distances > continuity_limit)
    discontinuity_column = column_edge_valid & (column_distances > continuity_limit)
    wrap_row = row_edge_valid & (row_distances > wrap_limit)
    wrap_column = column_edge_valid & (column_distances > wrap_limit)

    normal_row_valid = normals.valid[:-1] & normals.valid[1:]
    normal_column_valid = normals.valid[:, :-1] & normals.valid[:, 1:]
    normal_row_dot = np.sum(
        normals.positive_xyz[:-1] * normals.positive_xyz[1:], axis=-1
    )
    normal_column_dot = np.sum(
        normals.positive_xyz[:, :-1] * normals.positive_xyz[:, 1:], axis=-1
    )
    abrupt_row = normal_row_valid & (normal_row_dot < 0.0)
    abrupt_column = normal_column_valid & (normal_column_dot < 0.0)

    quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    p00, p01 = physical[:-1, :-1], physical[:-1, 1:]
    p10, p11 = physical[1:, :-1], physical[1:, 1:]
    top = np.linalg.norm(p01 - p00, axis=-1)
    bottom = np.linalg.norm(p11 - p10, axis=-1)
    left = np.linalg.norm(p10 - p00, axis=-1)
    right = np.linalg.norm(p11 - p01, axis=-1)
    edges = np.stack((top, bottom, left, right), axis=-1)
    minimum_edge = np.min(edges, axis=-1)
    maximum_edge = np.max(edges, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        distortion = maximum_edge / minimum_edge
    triangle_one = np.cross(p01 - p00, p10 - p00)
    triangle_two = np.cross(p11 - p10, p11 - p01)
    magnitude_one = np.linalg.norm(triangle_one, axis=-1)
    magnitude_two = np.linalg.norm(triangle_two, axis=-1)
    area_um2 = 0.5 * (magnitude_one + magnitude_two)
    area_scale = (
        median_distance * median_distance if math.isfinite(median_distance) else 1.0
    )
    degenerate_limit = max(np.finfo(np.float64).eps, area_scale * 1e-8)
    degenerate = quad_valid & (
        ~np.isfinite(area_um2)
        | ~np.isfinite(distortion)
        | (area_um2 <= degenerate_limit)
        | (minimum_edge <= math.sqrt(degenerate_limit))
    )
    denominator = magnitude_one * magnitude_two
    with np.errstate(divide="ignore", invalid="ignore"):
        triangle_cosine = np.sum(triangle_one * triangle_two, axis=-1) / denominator
    folded = quad_valid & ~degenerate & (triangle_cosine < 0.0)
    distorted = quad_valid & (distortion > metric_distortion_limit)
    quad_area_cm2 = np.where(quad_valid, area_um2 / 100_000_000.0, 0.0)

    reasons: dict[str, NDArray[np.bool_]] = {
        "invalid_vertex": ~valid,
        "out_of_bounds_vertex": valid & ~bounds,
        "normal_invalid_vertex": valid & ~normals.valid,
        "offset_out_vertex": normals.valid & ~offset_safe,
    }
    for name, row_edges, column_edges in (
        ("discontinuity_edge", discontinuity_row, discontinuity_column),
        ("neighbor_wrap_edge", wrap_row, wrap_column),
        ("abrupt_normal_flip_edge", abrupt_row, abrupt_column),
    ):
        marked = np.zeros(surface.shape, dtype=bool)
        _mark_row_edges(marked, row_edges)
        _mark_column_edges(marked, column_edges)
        reasons[name] = marked
    for name, quads in (("degenerate_quad", degenerate), ("folded_quad", folded)):
        marked = np.zeros(surface.shape, dtype=bool)
        _mark_quads(marked, quads)
        reasons[name] = marked

    base_defect = np.logical_or.reduce(tuple(reasons.values()))
    forbidden = dilate_one_cell(base_defect)
    summary = {
        "surface_shape": list(surface.shape),
        "strict_validity_matches_loader": strict_validity_matches_loader,
        "explicitly_masked_finite_nonnegative_coordinate_count": (
            explicitly_masked_coordinate_count
        ),
        "median_neighbor_distance_um": median_distance,
        "continuity_limit_um": continuity_limit,
        "neighbor_wrap_limit_um": wrap_limit,
        "raw_reason_vertex_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in reasons.items()
        },
        "raw_defect_vertex_count": int(np.count_nonzero(base_defect)),
        "one_cell_dilated_forbidden_vertex_count": int(np.count_nonzero(forbidden)),
        "known_clean_vertex_count": int(np.count_nonzero(~forbidden)),
        "distorted_quad_count": int(np.count_nonzero(distorted)),
        "localization_limitations": {
            "self_intersections": (
                "not localized by native_surface_sampler; never assumed zero; "
                "a positive preflight therefore requires later native/full validation"
            ),
            "resolution": (
                "stored TIFFXYZ control-grid normals; not calibrated as pixel-equivalent "
                "to the official full-resolution renderer"
            ),
        },
    }
    return GeometryMasks(
        forbidden=forbidden,
        base_defect=base_defect,
        sample_valid=sample_valid,
        quad_area_cm2=quad_area_cm2,
        distorted_quad=distorted,
        normals_xyz=normals.positive_xyz,
        normal_valid=normals.valid,
        summary=summary,
    )


def _transpose_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    row0, row1, column0, column1 = box
    return column0, column1, row0, row1


def largest_area_clean_rectangle(
    forbidden: NDArray[np.bool_],
    quad_area_cm2: NDArray[np.float64],
    *,
    min_area_cm2: float,
    max_area_cm2: float,
    minimum_side_vertices: int = 2,
) -> Mapping[str, Any]:
    """Exhaustively find the largest clean rectangle in the inclusive area band."""

    bad = np.asarray(forbidden, dtype=bool)
    areas = np.asarray(quad_area_cm2, dtype=np.float64)
    transposed = bad.shape[0] > bad.shape[1]
    if transposed:
        bad, areas = bad.T, areas.T
    height, width = bad.shape
    best: dict[str, Any] | None = None
    feasible_count = 0
    bands = 0
    for row0 in range(height - minimum_side_vertices + 1):
        blocked = np.zeros(width, dtype=bool)
        area_by_column = np.zeros(max(width - 1, 0), dtype=np.float64)
        for row1 in range(row0 + 1, height + 1):
            blocked |= bad[row1 - 1]
            if row1 - row0 < minimum_side_vertices:
                continue
            area_by_column += areas[row1 - 2]
            bands += 1
            area_prefix = np.pad(np.cumsum(area_by_column, dtype=np.float64), (1, 0))
            for run0, run1 in _runs(~blocked):
                if run1 - run0 < minimum_side_vertices:
                    continue
                for column0 in range(run0, run1 - minimum_side_vertices + 1):
                    first_q = column0 + minimum_side_vertices - 1
                    for q in range(first_q, run1):
                        area = float(area_prefix[q] - area_prefix[column0])
                        if area < min_area_cm2:
                            continue
                        if area > max_area_cm2:
                            break
                        feasible_count += 1
                        box = (row0, row1, column0, q + 1)
                        original_box = _transpose_box(box) if transposed else box
                        candidate = {
                            "window_r0_r1_c0_c1": list(original_box),
                            "shape": [
                                original_box[1] - original_box[0],
                                original_box[3] - original_box[2],
                            ],
                            "area_cm2": area,
                        }
                        if best is None or (area, tuple(-x for x in original_box)) > (
                            float(best["area_cm2"]),
                            tuple(-int(x) for x in best["window_r0_r1_c0_c1"]),
                        ):
                            best = candidate
    return {
        "search_space": (
            "all half-open axis-aligned rectangles with at least "
            f"{minimum_side_vertices}x{minimum_side_vertices} stored-grid vertices"
        ),
        "area_interval_inclusive_cm2": [min_area_cm2, max_area_cm2],
        "area_method": "float64 sum of two-triangle physical quad areas",
        "transposed_for_search": transposed,
        "row_bands_examined": bands,
        "feasible_rectangle_count": feasible_count,
        "largest_feasible_rectangle": best,
    }


def _rectangle_metrics(
    box: tuple[int, int, int, int],
    supported: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    area: NDArray[np.float64],
) -> dict[str, Any]:
    row0, row1, column0, column1 = box
    support_count = int(np.count_nonzero(supported[row0:row1, column0:column1]))
    valid_count = int(np.count_nonzero(valid[row0:row1, column0:column1]))
    # Match the optimizer's canonical float64 band-then-prefix difference.
    area_by_column = area[row0 : row1 - 1].sum(axis=0, dtype=np.float64)
    area_prefix = np.pad(np.cumsum(area_by_column, dtype=np.float64), (1, 0))
    area_cm2 = float(area_prefix[column1 - 1] - area_prefix[column0])
    return {
        "window_r0_r1_c0_c1": list(box),
        "shape": [row1 - row0, column1 - column0],
        "area_cm2": area_cm2,
        "valid_vertex_count": valid_count,
        "supported_vertex_count": support_count,
        "support_fraction": support_count / valid_count if valid_count else 0.0,
    }


def exact_max_support_rectangle(
    forbidden: NDArray[np.bool_],
    quad_area_cm2: NDArray[np.float64],
    supported: NDArray[np.bool_],
    valid: NDArray[np.bool_],
    *,
    min_area_cm2: float,
    max_area_cm2: float,
    minimum_side_vertices: int = 2,
    maximum_iterations: int = 64,
) -> Mapping[str, Any]:
    """Exact rational Dinkelbach maximum with an O(min(H,W)^2 max(H,W)) scan."""

    bad = np.asarray(forbidden, dtype=bool)
    area = np.asarray(quad_area_cm2, dtype=np.float64)
    support = np.asarray(supported, dtype=bool)
    denominator_valid = np.asarray(valid, dtype=bool)
    original_area = area
    original_support = support
    original_valid = denominator_valid
    transposed = bad.shape[0] > bad.shape[1]
    if transposed:
        bad, area, support, denominator_valid = (
            bad.T,
            area.T,
            support.T,
            denominator_valid.T,
        )
    height, width = bad.shape
    numerator, denominator = 0, 1
    iterations: list[dict[str, Any]] = []
    winning_box: tuple[int, int, int, int] | None = None

    for iteration in range(1, maximum_iterations + 1):
        best_delta: int | None = None
        best_scan_box: tuple[int, int, int, int] | None = None
        best_counts = (0, 0)
        best_area = -1.0
        support_by_column = np.zeros(width, dtype=np.int64)
        valid_by_column = np.zeros(width, dtype=np.int64)

        for row0 in range(height - minimum_side_vertices + 1):
            blocked = np.zeros(width, dtype=bool)
            area_by_column = np.zeros(max(width - 1, 0), dtype=np.float64)
            support_by_column.fill(0)
            valid_by_column.fill(0)
            for row1 in range(row0 + 1, height + 1):
                blocked |= bad[row1 - 1]
                support_by_column += support[row1 - 1].astype(np.int64)
                valid_by_column += denominator_valid[row1 - 1].astype(np.int64)
                if row1 - row0 < minimum_side_vertices:
                    continue
                area_by_column += area[row1 - 2]
                area_prefix = np.pad(
                    np.cumsum(area_by_column, dtype=np.float64), (1, 0)
                )
                transformed = (
                    denominator * support_by_column - numerator * valid_by_column
                )
                transformed_prefix = np.pad(
                    np.cumsum(transformed, dtype=np.int64), (1, 0)
                )
                support_prefix = np.pad(
                    np.cumsum(support_by_column, dtype=np.int64), (1, 0)
                )
                valid_prefix = np.pad(
                    np.cumsum(valid_by_column, dtype=np.int64), (1, 0)
                )

                for run0, run1 in _runs(~blocked):
                    if run1 - run0 < minimum_side_vertices:
                        continue
                    candidates: deque[int] = deque()
                    next_add = run0
                    for column1 in range(run0 + minimum_side_vertices, run1 + 1):
                        area_end = area_prefix[column1 - 1]
                        maximum_left = column1 - minimum_side_vertices
                        while (
                            next_add <= maximum_left
                            and area_end - area_prefix[next_add] >= min_area_cm2
                        ):
                            while (
                                candidates
                                and transformed_prefix[candidates[-1]]
                                > transformed_prefix[next_add]
                            ):
                                candidates.pop()
                            candidates.append(next_add)
                            next_add += 1
                        while (
                            candidates
                            and area_end - area_prefix[candidates[0]] > max_area_cm2
                        ):
                            candidates.popleft()
                        if not candidates:
                            continue
                        column0 = candidates[0]
                        candidate_area = float(area_end - area_prefix[column0])
                        if not (min_area_cm2 <= candidate_area <= max_area_cm2):
                            continue
                        support_count = int(
                            support_prefix[column1] - support_prefix[column0]
                        )
                        valid_count = int(valid_prefix[column1] - valid_prefix[column0])
                        if valid_count <= 0:
                            continue
                        delta = denominator * support_count - numerator * valid_count
                        scan_box = (row0, row1, column0, column1)
                        original_box = (
                            _transpose_box(scan_box) if transposed else scan_box
                        )
                        choose = best_delta is None or delta > best_delta
                        if delta == best_delta:
                            left_ratio = support_count * best_counts[1]
                            right_ratio = best_counts[0] * valid_count
                            choose = left_ratio > right_ratio or (
                                left_ratio == right_ratio
                                and (
                                    candidate_area > best_area
                                    or (
                                        candidate_area == best_area
                                        and original_box
                                        < (
                                            _transpose_box(best_scan_box)
                                            if transposed
                                            else best_scan_box
                                        )  # type: ignore[arg-type]
                                    )
                                )
                            )
                        if choose:
                            best_delta = delta
                            best_scan_box = scan_box
                            best_counts = (support_count, valid_count)
                            best_area = candidate_area

        if best_scan_box is None or best_delta is None:
            return {
                "algorithm": "exact rational Dinkelbach + monotone band/deque scan",
                "transposed_for_search": transposed,
                "iterations": iterations,
                "best_rectangle": None,
            }
        original_box = _transpose_box(best_scan_box) if transposed else best_scan_box
        iterations.append(
            {
                "iteration": iteration,
                "ratio_numerator": numerator,
                "ratio_denominator": denominator,
                "maximum_transformed_delta": best_delta,
                "winner": list(original_box),
            }
        )
        winning_box = original_box
        if best_delta == 0:
            break
        divisor = math.gcd(best_counts[0], best_counts[1])
        numerator = best_counts[0] // divisor
        denominator = best_counts[1] // divisor
    else:
        raise RuntimeError("exact maximum-support search did not converge")

    assert winning_box is not None
    metrics = _rectangle_metrics(
        winning_box, original_support, original_valid, original_area
    )
    return {
        "algorithm": "exact rational Dinkelbach + monotone band/deque scan",
        "objective": "maximum supported/valid vertex fraction",
        "tie_break": "maximum area, then lexicographically smallest original box",
        "transposed_for_search": transposed,
        "iterations": iterations,
        "optimality_certificate_maximum_transformed_delta": 0,
        "best_rectangle": metrics,
    }


class ChunkGroupedSampler:
    """Trilinear sampler fetching every source chunk at most once per call."""

    def __init__(self, opened: OpenedVolume) -> None:
        self.volume = opened.volume_zyx
        if self.volume.chunks is None:
            raise ValueError("Zarr array must expose chunk dimensions")
        self.shape_zyx = tuple(int(value) for value in self.volume.shape)
        self.chunks_zyx = tuple(int(value) for value in self.volume.chunks)
        self.roi_records: list[dict[str, Any]] = []
        self.requested_point_count = 0

    def sample(
        self,
        coordinates_xyz: NDArray[np.float64],
        *,
        point_valid: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        coordinates = np.asarray(coordinates_xyz, dtype=np.float64)
        requested = np.asarray(point_valid, dtype=bool)
        point_shape = coordinates.shape[:-1]
        flat = coordinates.reshape(-1, 3)
        flat_requested = requested.reshape(-1)
        z_size, y_size, x_size = self.shape_zyx
        finite = np.isfinite(flat).all(axis=1)
        in_bounds = (
            finite
            & (flat[:, 0] >= 0.0)
            & (flat[:, 0] <= x_size - 1)
            & (flat[:, 1] >= 0.0)
            & (flat[:, 1] <= y_size - 1)
            & (flat[:, 2] >= 0.0)
            & (flat[:, 2] <= z_size - 1)
        )
        usable = flat_requested & in_bounds
        output = np.zeros(flat.shape[0], dtype=np.float32)
        output_valid = np.zeros(flat.shape[0], dtype=bool)
        indices = np.flatnonzero(usable)
        self.requested_point_count += int(indices.size)
        if not indices.size:
            return output.reshape(point_shape), output_valid.reshape(point_shape)
        selected_xyz = flat[indices]
        lower_xyz = np.floor(selected_xyz).astype(np.int64)
        upper_xyz = np.minimum(
            lower_xyz + 1,
            np.asarray((x_size - 1, y_size - 1, z_size - 1), dtype=np.int64),
        )
        fraction_xyz = selected_xyz - lower_xyz
        point_rows: list[NDArray[np.int64]] = []
        voxel_rows: list[NDArray[np.int64]] = []
        weight_rows: list[NDArray[np.float64]] = []
        point_index = np.arange(indices.size, dtype=np.int64)
        for z_side in (0, 1):
            for y_side in (0, 1):
                for x_side in (0, 1):
                    sides = np.asarray((x_side, y_side, z_side), dtype=np.int64)
                    corner_xyz = np.where(sides[None, :] == 0, lower_xyz, upper_xyz)
                    component_weight = np.where(
                        sides[None, :] == 0, 1.0 - fraction_xyz, fraction_xyz
                    )
                    weights = np.prod(component_weight, axis=1)
                    contributes = weights > 0.0
                    point_rows.append(point_index[contributes])
                    voxel_rows.append(corner_xyz[contributes, ::-1])
                    weight_rows.append(weights[contributes])
        corner_points = np.concatenate(point_rows)
        corner_zyx = np.concatenate(voxel_rows)
        corner_weights = np.concatenate(weight_rows)
        chunks = np.asarray(self.chunks_zyx, dtype=np.int64)
        chunk_ids = corner_zyx // chunks
        unique, inverse = np.unique(chunk_ids, axis=0, return_inverse=True)
        shape_array = np.asarray(self.shape_zyx, dtype=np.int64)

        def load_chunk(
            chunk_id: NDArray[np.int64],
        ) -> tuple[NDArray[np.int64], NDArray[Any]]:
            start = chunk_id * chunks
            stop = np.minimum(start + chunks, shape_array)
            slices = tuple(
                slice(int(left), int(right)) for left, right in zip(start, stop)
            )
            chunk = np.asarray(self.volume[slices])
            expected = tuple(int(right - left) for left, right in zip(start, stop))
            if chunk.shape != expected:
                raise ValueError(
                    f"Zarr chunk read returned {chunk.shape}, expected {expected}"
                )
            return start, chunk

        workers = min(8, max(1, len(unique)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(load_chunk, [row for row in unique]))
        accumulated = np.zeros(indices.size, dtype=np.float64)
        for group_index, (chunk_id, (chunk_start, chunk)) in enumerate(
            zip(unique, loaded)
        ):
            entries = np.flatnonzero(inverse == group_index)
            local_zyx = corner_zyx[entries] - chunk_start
            values = np.asarray(
                chunk[local_zyx[:, 0], local_zyx[:, 1], local_zyx[:, 2]],
                dtype=np.float64,
            )
            np.add.at(
                accumulated,
                corner_points[entries],
                corner_weights[entries] * values,
            )
            self.roi_records.append(
                {
                    "chunk_zyx": chunk_id.astype(int).tolist(),
                    "chunk_start_zyx": chunk_start.astype(int).tolist(),
                    "chunk_stop_zyx": (chunk_start + np.asarray(chunk.shape))
                    .astype(int)
                    .tolist(),
                    "sha256_decompressed": _sha256_array(chunk),
                }
            )
        output[indices] = accumulated.astype(np.float32)
        output_valid[indices] = True
        return output.reshape(point_shape), output_valid.reshape(point_shape)

    def manifest(self) -> Mapping[str, Any]:
        rows = sorted(
            f"{record['chunk_zyx']}:{record['sha256_decompressed']}"
            for record in self.roi_records
        )
        digest = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
        return {
            "grouping": "all eight trilinear corners grouped by source chunk",
            "roi_rule": "each unique source chunk fetched once per sample call",
            "shape_zyx": list(self.shape_zyx),
            "chunks_zyx": list(self.chunks_zyx),
            "requested_valid_point_count": self.requested_point_count,
            "roi_read_count": len(self.roi_records),
            "decompressed_roi_hash_aggregate_sha256": digest,
        }


def sample_offsets(
    sampler: ChunkGroupedSampler,
    surface: SurfaceGrid,
    normals_xyz: NDArray[np.float64],
    point_valid: NDArray[np.bool_],
    offsets_voxels: Sequence[int],
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    offsets = np.asarray(offsets_voxels, dtype=np.float64)
    coordinates = (
        surface.points_xyz[None, ...]
        + offsets[:, None, None, None] * normals_xyz[None, ...]
    )
    validity = np.broadcast_to(point_valid, coordinates.shape[:-1])
    values, sampled_valid = sampler.sample(coordinates, point_valid=validity)
    # Match tifxyz_render_pipeline._cast_output_frame for uint8 output before >0.
    quantized = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    return quantized, sampled_valid


def foreground_runs(mask: NDArray[np.bool_]) -> int:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    return int(np.count_nonzero(np.diff(padded) == 1))


def verdict_exit_code(verdict: str) -> int:
    """Return an automation-safe exit code for a completed preflight verdict."""

    codes = {"preflight_pass": 0, "fail": 2, "inconclusive": 3}
    if verdict not in codes:
        raise ValueError(f"unsupported preflight verdict: {verdict!r}")
    return codes[verdict]


def evaluate_transects(
    frames: NDArray[np.uint8],
    box: Sequence[int],
    *,
    seed: int,
    sample_count: int = 50,
) -> Mapping[str, Any]:
    row0, row1, column0, column1 = (int(value) for value in box)
    vertex_count = (row1 - row0) * (column1 - column0)
    if vertex_count < sample_count:
        return {
            "evaluated": False,
            "reason": f"rectangle has {vertex_count} vertices, fewer than {sample_count}",
            "pass": False,
        }
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(vertex_count, size=sample_count, replace=False))
    local = frames[:, row0:row1, column0:column1].reshape(frames.shape[0], -1) > 0
    center = frames.shape[0] // 2
    counts = {"single_run_centered": 0, "zero_run": 0, "multi_run": 0, "off_center": 0}
    for index in chosen:
        trace = local[:, index]
        run_count = foreground_runs(trace)
        if run_count == 0:
            counts["zero_run"] += 1
        elif run_count > 1:
            counts["multi_run"] += 1
        elif not trace[center]:
            counts["off_center"] += 1
        else:
            counts["single_run_centered"] += 1
    return {
        "evaluated": True,
        "rng": "numpy.default_rng",
        "rng_seed": seed,
        "sample_count": sample_count,
        "chosen_flat_indices_local": chosen.astype(int).tolist(),
        "category_counts": counts,
        "pass_count": counts["single_run_centered"],
        "pass": counts["single_run_centered"] == sample_count,
    }


def run_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.monotonic()
    asset = load_tifxyz_asset(args.tifxyz, resolution="stored")
    opened = open_volume_zyx(
        args.m7_zarr,
        array_path=args.m7_array_path,
        axes="zyx",
    )
    if tuple(opened.volume_zyx.shape) != tuple(args.volume_shape_zyx):
        raise ValueError(
            f"m7 shape {opened.volume_zyx.shape} differs from expected {args.volume_shape_zyx}"
        )
    masks = derive_geometry_masks(
        asset.surface,
        volume_shape_zyx=opened.volume_zyx.shape,
        voxel_um=args.voxel_um,
        maximum_offset_voxels=15,
    )
    offsets_um = [value * args.voxel_um for value in DEFAULT_TRANSECT_OFFSETS]
    native_reports = {
        label: validate_surface_geometry(
            asset.surface,
            volume_shape_zyx=opened.volume_zyx.shape,
            voxel_size_zyx_um=(args.voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }
    geometry_search = largest_area_clean_rectangle(
        masks.forbidden,
        masks.quad_area_cm2,
        min_area_cm2=args.min_area_cm2,
        max_area_cm2=args.max_area_cm2,
    )
    sampler = ChunkGroupedSampler(opened)
    m7_search: Mapping[str, Any]
    permissive_upper_bound_search: Mapping[str, Any]
    transects: Mapping[str, Any]
    best_geometry = geometry_search["largest_feasible_rectangle"]
    # The terminal-rejection calculation intentionally uses a permissive
    # superset.  It forbids only per-vertex failures that can never be repaired
    # by cropping.  Valid vertices without a usable normal/sample are counted as
    # supported optimistically.  Fold/edge/dilation/self-intersection masks are
    # *not* applied, so its maximum is an upper bound on any feasible crop.
    support_frames, support_valid = sample_offsets(
        sampler,
        asset.surface,
        masks.normals_xyz,
        masks.sample_valid,
        DEFAULT_SUPPORT_OFFSETS,
    )
    if not np.all(support_valid[:, masks.sample_valid]):
        raise ValueError("one or more requested ±2 samples were invalid")
    supported = (support_frames > 0).any(axis=0) & masks.sample_valid
    upper_forbidden = asset.surface.valid & masks.normal_valid & ~masks.sample_valid
    optimistic_supported = supported | (asset.surface.valid & ~masks.sample_valid)
    permissive_upper_bound_search = exact_max_support_rectangle(
        upper_forbidden,
        masks.quad_area_cm2,
        optimistic_supported,
        asset.surface.valid,
        min_area_cm2=args.min_area_cm2,
        max_area_cm2=args.max_area_cm2,
    )
    upper_best = permissive_upper_bound_search.get("best_rectangle")
    upper_fraction = float(upper_best["support_fraction"]) if upper_best else 0.0

    if best_geometry is None:
        m7_search = {
            "evaluated": False,
            "reason": "no known-clean rectangle in the mandatory area interval",
        }
        transects = {
            "evaluated": False,
            "reason": "strict m7 support gate not reached",
            "pass": False,
        }
    else:
        m7_search = exact_max_support_rectangle(
            masks.forbidden,
            masks.quad_area_cm2,
            supported,
            masks.sample_valid,
            min_area_cm2=args.min_area_cm2,
            max_area_cm2=args.max_area_cm2,
        )
        best = m7_search.get("best_rectangle")
        if best is None or float(best["support_fraction"]) < args.min_support:
            transects = {
                "evaluated": False,
                "reason": "exact maximum ±2 support is below the middle gate",
                "pass": False,
            }
        else:
            all_frames, all_valid = sample_offsets(
                sampler,
                asset.surface,
                masks.normals_xyz,
                masks.sample_valid,
                DEFAULT_TRANSECT_OFFSETS,
            )
            row0, row1, column0, column1 = best["window_r0_r1_c0_c1"]
            if not np.all(all_valid[:, row0:row1, column0:column1]):
                raise ValueError(
                    "one or more selected 31-offset transect samples were invalid"
                )
            transects = evaluate_transects(
                all_frames,
                best["window_r0_r1_c0_c1"],
                seed=args.seed,
            )
    passed = bool(transects.get("pass", False))
    if upper_fraction < args.min_support:
        verdict = "fail"
        verdict_reason = "permissive_m7_support_upper_bound_below_threshold"
    elif passed:
        verdict = "preflight_pass"
        verdict_reason = "stored_grid_preflight_pass_requires_full_native_validation"
    else:
        verdict = "inconclusive"
        verdict_reason = (
            "permissive_upper_bound_not_rejected_but_strict_pipeline_did_not_pass"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_utc": _utc_now(),
        "candidate_id": args.candidate_id,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "evidence_scope": (
            "stored-grid free-local falsification preflight; a pass is non-terminal and "
            "requires full-resolution official rendering plus self-intersection validation"
        ),
        "inputs": {
            "tifxyz": asset.manifest(),
            "m7": opened.manifest(),
        },
        "algorithm_and_gates": {
            "coordinate_order": "TIFFXYZ xyz; volume zyx",
            "surface_resolution": "stored",
            "voxel_size_um": args.voxel_um,
            "geometry_defect_margin": "8-connected one stored-grid cell",
            "known_local_hard_defects": [
                "invalid",
                "out-of-bounds",
                "normal-invalid",
                "±15-offset-out",
                "continuity",
                "neighbor-wrap",
                "degenerate",
                "folded",
                "normal-flip",
            ],
            "self_intersection_policy": "unlocalized/unknown; never fabricated as zero",
            "rectangle_area_cm2_inclusive": [args.min_area_cm2, args.max_area_cm2],
            "distortion_note": "reported, not hard-masked; canonical allowance is <=5%",
            "m7_support_offsets_voxels": list(DEFAULT_SUPPORT_OFFSETS),
            "m7_quantization": "rint clip[0,255] uint8, then >0",
            "minimum_m7_support_fraction": args.min_support,
            "transect_offsets_voxels": list(DEFAULT_TRANSECT_OFFSETS),
            "transect_rule": "50/50 exactly one foreground run containing offset 0",
            "rng_seed": args.seed,
            "orientations": "co-equal; centered symmetric stacks are exact reversals",
        },
        "geometry": {
            "localized_masks": masks.summary,
            "native_reports_by_orientation": native_reports,
            "exhaustive_clean_rectangle_search": geometry_search,
        },
        "m7_support_search": m7_search,
        "m7_permissive_upper_bound_search": {
            "purpose": (
                "terminal-rejection-safe upper bound: no fold/edge/dilation/"
                "self-intersection exclusions; unsampleable valid vertices are optimistic positives"
            ),
            "forbidden_rule": (
                "only valid base-coordinate out-of-volume or normal-valid ±15 endpoint out-of-volume"
            ),
            "optimistic_unsampleable_valid_vertex_count": int(
                np.count_nonzero(asset.surface.valid & ~masks.sample_valid)
            ),
            "result": permissive_upper_bound_search,
        },
        "transects": transects,
        "zarr_sampling": sampler.manifest(),
        "elapsed_seconds": time.monotonic() - started,
        "paid_compute_cost_usd": 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--tifxyz", required=True)
    parser.add_argument("--m7-zarr", default=DEFAULT_M7)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--min-area-cm2", type=float, default=0.5)
    parser.add_argument("--max-area-cm2", type=float, default=4.0)
    parser.add_argument("--min-support", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1447)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_preflight(args)
    except Exception as error:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "created_utc": _utc_now(),
            "candidate_id": args.candidate_id,
            "verdict": "fail",
            "verdict_reason": "preflight_error_fail_closed",
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs": {
                "tifxyz": sanitize_source(args.tifxyz),
                "m7_zarr": sanitize_source(args.m7_zarr),
            },
            "paid_compute_cost_usd": 0.0,
        }
        _atomic_json(args.output, result)
        print(json.dumps(_json_safe(result), sort_keys=True), file=sys.stderr)
        return 1
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "candidate_id": result["candidate_id"],
                "verdict": result["verdict"],
                "verdict_reason": result["verdict_reason"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return verdict_exit_code(str(result["verdict"]))


if __name__ == "__main__":
    raise SystemExit(main())
