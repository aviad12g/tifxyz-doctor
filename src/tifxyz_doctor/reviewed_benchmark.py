"""Helpers for a reviewed same-wrap control and synthetic switch benchmark.

The public PHercParis4 spiral corpus contains TIFXYZ patches derived from
human-reviewed ``same_wrap`` point collections.  Those annotations do not prove
that every cell in a patch is globally correct.  They do, however, provide a
useful negative-control corridor: the annotated path is intended to remain on
one winding.

Synthetic cases in this module keep the observed surface geometry and move one
side of a deterministic seam along locally averaged surface normals.  They are
controlled normal-offset proxies, not labels for naturally occurring tracing
failures.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .io import TifxyzData
from .topology import valid_quad_mask


@dataclass(frozen=True)
class SyntheticSwitch:
    """One deterministic normal-offset perturbation and its labeled seam."""

    data: TifxyzData
    seam_cells: np.ndarray
    evaluation_cells: np.ndarray
    orientation: str
    seam_index: int
    offset_voxels: float
    transition_width_cells: int


def patch_overlap_components(
    overlap_graph: dict[str, Any],
    patch_ids: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, ...]]]:
    """Return overlap edges and connected components for selected patches."""

    adjacency = {patch_id: set() for patch_id in patch_ids}
    edges: set[tuple[str, str]] = set()
    collections = overlap_graph.get("collections", {})
    if not isinstance(collections, dict):
        raise ValueError("overlap graph collections must be an object")
    prefix = "between_patches__"
    for collection in collections.values():
        if not isinstance(collection, dict):
            continue
        name = collection.get("name")
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        endpoints = name[len(prefix) :].split("__")
        if len(endpoints) != 2:
            continue
        first, second = endpoints
        if first not in patch_ids or second not in patch_ids or first == second:
            continue
        edge = tuple(sorted((first, second)))
        edges.add(edge)
        adjacency[first].add(second)
        adjacency[second].add(first)

    remaining = set(patch_ids)
    components: list[tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members, reverse=True))
        remaining -= members
        components.append(tuple(sorted(members)))
    components.sort()
    return sorted(edges), components


def overlap_isolated_split(
    overlap_graph: dict[str, Any],
    patch_ids: set[str],
    development_ids: set[str],
    *,
    holdout_target_patches: int,
    salt: str,
) -> dict[str, Any]:
    """Build a component-isolated development/holdout split."""

    if not development_ids <= patch_ids:
        missing = sorted(development_ids - patch_ids)
        raise ValueError(f"development patches are absent: {missing[:3]}")
    if holdout_target_patches < 1:
        raise ValueError("holdout target must be positive")
    edges, components = patch_overlap_components(overlap_graph, patch_ids)
    contaminated = [
        component
        for component in components
        if development_ids.intersection(component)
    ]
    clean = [
        component
        for component in components
        if not development_ids.intersection(component)
    ]
    clean_ranked = sorted(
        clean,
        key=lambda component: (
            hashlib.sha256(
                (salt + "\0" + "\0".join(component)).encode()
            ).hexdigest(),
            component,
        ),
    )
    selected_holdout_components: list[tuple[str, ...]] = []
    selected_count = 0
    for component in clean_ranked:
        selected_holdout_components.append(component)
        selected_count += len(component)
        if selected_count >= holdout_target_patches:
            break
    contaminated_ids = {
        patch_id for component in contaminated for patch_id in component
    }
    clean_ids = {patch_id for component in clean for patch_id in component}
    holdout_ids = {
        patch_id
        for component in selected_holdout_components
        for patch_id in component
    }
    component_ids: dict[str, str] = {}
    for component in components:
        digest = hashlib.sha256("\0".join(component).encode()).hexdigest()
        component_id = f"sha256:{digest}"
        component_ids.update({patch_id: component_id for patch_id in component})
    return {
        "overlap_edges": [list(edge) for edge in edges],
        "components": [list(component) for component in components],
        "component_ids": component_ids,
        "development_ids": sorted(development_ids),
        "development_connected_ids": sorted(contaminated_ids),
        "development_related_excluded_ids": sorted(
            contaminated_ids - development_ids
        ),
        "clean_holdout_pool_ids": sorted(clean_ids),
        "selected_holdout_components": [
            list(component) for component in selected_holdout_components
        ],
        "selected_holdout_ids": sorted(holdout_ids),
    }


def dilate_cells(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 2-D boolean cell mask using Chebyshev distance."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {source.shape}")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    result = source.copy()
    height, width = source.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            src_r0, src_r1 = max(0, -dr), height - max(0, dr)
            src_c0, src_c1 = max(0, -dc), width - max(0, dc)
            dst_r0, dst_r1 = max(0, dr), height - max(0, -dr)
            dst_c0, dst_c1 = max(0, dc), width - max(0, -dc)
            if src_r1 <= src_r0 or src_c1 <= src_c0:
                continue
            result[dst_r0:dst_r1, dst_c0:dst_c1] |= source[
                src_r0:src_r1,
                src_c0:src_c1,
            ]
    return result


def same_wrap_annotation_neighborhoods(
    corr_points_results: dict[str, Any],
    vertex_shape: tuple[int, int],
    *,
    radius: int = 4,
) -> np.ndarray:
    """Map valid annotation samples to dilated cell-space neighborhoods.

    The public correlation result supplies mapped sample locations, not an
    ordered polyline. This function deliberately does not imply labels between
    samples.
    """
    height, width = (int(vertex_shape[0]), int(vertex_shape[1]))
    if height < 2 or width < 2:
        return np.zeros((max(0, height - 1), max(0, width - 1)), dtype=bool)
    seeds = np.zeros((height - 1, width - 1), dtype=bool)
    points = corr_points_results.get("points_list", [])
    if not isinstance(points, list):
        return seeds

    for point in points:
        if not isinstance(point, dict) or point.get("valid") is False:
            continue
        locations = point.get("model_locations", [])
        if not isinstance(locations, list):
            continue
        for location in locations:
            if not isinstance(location, dict):
                continue
            try:
                row = float(location["h"])
                col = float(location["w"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(row) or not np.isfinite(col):
                continue
            # A vertex location can touch up to four cells. Mark all of them
            # before dilation rather than assigning the path to one arbitrary
            # side of a cell boundary.
            for cell_row in (int(np.floor(row)) - 1, int(np.floor(row))):
                for cell_col in (int(np.floor(col)) - 1, int(np.floor(col))):
                    if 0 <= cell_row < height - 1 and 0 <= cell_col < width - 1:
                        seeds[cell_row, cell_col] = True
    return dilate_cells(seeds, radius)


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1)
    result = np.full(vectors.shape, np.nan, dtype=np.float64)
    good = np.isfinite(lengths) & (lengths > 1e-12)
    result[good] = vectors[good] / lengths[good, None]
    return result


def averaged_vertex_normals(coordinates: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Average the two official-split triangle normals onto grid vertices."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    vertex_valid = np.asarray(valid, dtype=bool)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected coordinates shaped (height, width, 3), got {xyz.shape}")
    if vertex_valid.shape != xyz.shape[:2]:
        raise ValueError(
            f"Validity mask shape {vertex_valid.shape} differs from coordinates {xyz.shape[:2]}"
        )

    normals = np.zeros(xyz.shape, dtype=np.float64)
    counts = np.zeros(vertex_valid.shape, dtype=np.int32)
    if xyz.shape[0] < 2 or xyz.shape[1] < 2:
        normals[:] = np.nan
        return normals

    p00 = xyz[:-1, :-1]
    p01 = xyz[:-1, 1:]
    p10 = xyz[1:, :-1]
    p11 = xyz[1:, 1:]
    cells = valid_quad_mask(vertex_valid)
    unit_a = _unit(np.cross(p01 - p00, p10 - p00))
    unit_b = _unit(np.cross(p11 - p01, p10 - p01))
    cell_normal = _unit(unit_a + unit_b)
    good_cells = cells & np.isfinite(cell_normal).all(axis=-1)
    contribution = np.where(good_cells[..., None], cell_normal, 0.0)

    for row_slice, col_slice in (
        (slice(None, -1), slice(None, -1)),
        (slice(None, -1), slice(1, None)),
        (slice(1, None), slice(None, -1)),
        (slice(1, None), slice(1, None)),
    ):
        normals[row_slice, col_slice] += contribution
        counts[row_slice, col_slice] += good_cells

    result = _unit(normals)
    result[(counts == 0) | ~vertex_valid] = np.nan
    return result


def choose_seam(valid: np.ndarray, normals: np.ndarray) -> tuple[str, int]:
    """Choose the best-supported central horizontal or vertical seam."""
    vertex_valid = np.asarray(valid, dtype=bool)
    normal_valid = np.isfinite(np.asarray(normals)).all(axis=-1)
    supported = vertex_valid & normal_valid
    cells = valid_quad_mask(supported)
    height, width = vertex_valid.shape
    if height < 4 or width < 4 or not cells.any():
        raise ValueError("Surface has no supported interior seam")

    candidates: list[tuple[int, str, int]] = []
    col_start, col_stop = max(1, width // 4), min(width - 1, (3 * width) // 4 + 1)
    for col in range(col_start, col_stop):
        candidates.append((int(cells[:, col - 1].sum()), "vertical", col))
    row_start, row_stop = max(1, height // 4), min(height - 1, (3 * height) // 4 + 1)
    for row in range(row_start, row_stop):
        candidates.append((int(cells[row - 1, :].sum()), "horizontal", row))
    support, orientation, index = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    if support < 4:
        raise ValueError(f"Best supported seam has only {support} valid cell(s)")
    return orientation, index


def _transition_coefficients(length: int, seam_index: int, width: int) -> np.ndarray:
    if width < 1:
        raise ValueError("transition width must be at least one cell")
    positions = np.arange(length, dtype=np.float64)
    if width == 1:
        return (positions >= seam_index).astype(np.float64)
    start = seam_index - width / 2.0
    t = np.clip((positions - start) / float(width), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def inject_normal_offset_switch(
    data: TifxyzData,
    *,
    offset_voxels: float,
    transition_width_cells: int,
    evaluation_radius: int = 1,
) -> SyntheticSwitch:
    """Create a controlled one-sided normal offset on a real TIFXYZ surface."""
    if offset_voxels < 0:
        raise ValueError("offset_voxels must be non-negative")
    normals = averaged_vertex_normals(data.coordinates, data.valid)
    orientation, seam_index = choose_seam(data.valid, normals)
    height, width = data.valid.shape
    axis_length = width if orientation == "vertical" else height
    axis_coefficients = _transition_coefficients(
        axis_length,
        seam_index,
        transition_width_cells,
    )
    coefficients = (
        np.broadcast_to(axis_coefficients[None, :], (height, width))
        if orientation == "vertical"
        else np.broadcast_to(axis_coefficients[:, None], (height, width))
    )

    if offset_voxels == 0:
        coordinates = np.asarray(data.coordinates).copy()
    else:
        coordinates = np.asarray(data.coordinates, dtype=np.float64).copy()
        movable = data.valid & np.isfinite(normals).all(axis=-1)
        coordinates[movable] += (
            float(offset_voxels) * coefficients[movable, None] * normals[movable]
        )
        coordinates = coordinates.astype(np.float32)

    cells = valid_quad_mask(data.valid)
    coefficient_span = np.maximum.reduce(
        (
            coefficients[:-1, :-1],
            coefficients[:-1, 1:],
            coefficients[1:, :-1],
            coefficients[1:, 1:],
        )
    ) - np.minimum.reduce(
        (
            coefficients[:-1, :-1],
            coefficients[:-1, 1:],
            coefficients[1:, :-1],
            coefficients[1:, 1:],
        )
    )
    seam_cells = cells & (coefficient_span > 1e-12)
    evaluation_cells = dilate_cells(seam_cells, evaluation_radius) & cells
    synthetic = TifxyzData(
        path=(
            data.path
            if offset_voxels == 0
            else Path(f"{data.path}__synthetic_switch")
        ),
        coordinates=coordinates,
        valid=data.valid.copy(),
        metadata=(
            dict(data.metadata)
            if offset_voxels == 0
            else {**data.metadata, "synthetic_switch_proxy": True}
        ),
        explicit_mask=(
            None if data.explicit_mask is None else data.explicit_mask.copy()
        ),
    )
    return SyntheticSwitch(
        data=synthetic,
        seam_cells=seam_cells,
        evaluation_cells=evaluation_cells,
        orientation=orientation,
        seam_index=seam_index,
        offset_voxels=float(offset_voxels),
        transition_width_cells=int(transition_width_cells),
    )
