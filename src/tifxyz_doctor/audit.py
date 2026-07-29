"""Deterministic TIFXYZ geometry audit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable

import numpy as np

from ._version import __version__
from .io import TifxyzData
from .topology import (
    enclosed_invalid_regions,
    label_components_4,
    label_components_8,
    valid_quad_mask,
)


@dataclass(frozen=True)
class AuditConfig:
    """Transparent review thresholds.

    These thresholds produce *review cues*, not assertions that a surface is
    physically wrong. Format-integrity findings are reported separately.
    """

    expected_spacing_x: float | None = None
    expected_spacing_y: float | None = None
    long_edge_ratio: float = 2.0
    short_edge_ratio: float = 0.5
    normal_jump_degrees: float = 75.0
    condition_number: float = 4.0
    symmetric_stretch: float = 2.0
    symmetric_dirichlet: float = 20.0
    area_ratio_low: float = 0.25
    area_ratio_high: float = 4.0
    shear: float = math.cos(math.radians(30.0))
    normal_step_ratio: float = 0.25
    normal_step_min_component_cells: int = 8
    nonlocal_distance_ratio: float = 0.25
    nonlocal_uv_exclusion: int = 4
    max_proximity_points: int = 100_000
    max_proximity_pairs: int = 10_000


def _finite_values(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    quantiles = np.percentile(finite, [5, 50, 95])
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(finite.max()),
    }


def _top_locations(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    limit: int = 50,
    largest: bool = True,
) -> list[dict[str, Any]]:
    """Return deterministic top grid locations for an already-defined cue."""
    selected = np.argwhere(np.asarray(mask, dtype=bool))
    if selected.size == 0 or limit <= 0:
        return []
    selected_values = np.asarray(values)[tuple(selected.T)].astype(np.float64)
    order = np.argsort(selected_values, kind="stable")
    if largest:
        order = order[::-1]
    records: list[dict[str, Any]] = []
    for index in order[:limit]:
        row, col = selected[int(index)]
        raw_value = float(selected_values[int(index)])
        serialized_value: float | None = raw_value if math.isfinite(raw_value) else None
        records.append(
            {
                "rc": [int(row), int(col)],
                "value": serialized_value,
                **(
                    {"nonfinite": "positive_infinity" if raw_value > 0 else "negative_infinity"}
                    if math.isinf(raw_value)
                    else ({"nonfinite": "nan"} if math.isnan(raw_value) else {})
                ),
            }
        )
    return records


def _region_summaries(labels: np.ndarray, *, limit: int = 100) -> list[dict[str, Any]]:
    """Summarize labeled regions largest-first without exposing full arrays."""
    records: list[dict[str, Any]] = []
    for label in range(1, int(labels.max(initial=0)) + 1):
        coordinates = np.argwhere(labels == label)
        if coordinates.size == 0:
            continue
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        centroid = coordinates.mean(axis=0)
        records.append(
            {
                "label": label,
                "cell_count": int(coordinates.shape[0]),
                "bbox_rc": [
                    [int(minimum[0]), int(minimum[1])],
                    [int(maximum[0]), int(maximum[1])],
                ],
                "centroid_rc": [float(centroid[0]), float(centroid[1])],
            }
        )
    records.sort(key=lambda item: (-int(item["cell_count"]), int(item["label"])))
    return records[:limit]


def _safe_unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(vectors, axis=-1)
    units = np.full(vectors.shape, np.nan, dtype=np.float64)
    good = np.isfinite(lengths) & (lengths > 1e-12)
    units[good] = vectors[good] / lengths[good, None]
    return units, lengths


def _metadata_bbox(metadata: dict[str, Any]) -> np.ndarray | None:
    raw = metadata.get("bbox")
    try:
        parsed = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if parsed.shape != (2, 3) or not np.isfinite(parsed).all():
        return None
    return parsed


def _ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return None
    return numerator / denominator


def _spacing_reference(declared: float | None, measured: np.ndarray) -> float | None:
    """Choose a positive finite spacing without emitting all-NaN warnings."""
    if declared is not None and math.isfinite(declared) and declared > 0:
        return float(declared)
    finite = _finite_values(measured)
    finite = finite[finite > 1e-12]
    return float(np.median(finite)) if finite.size else None


def _validate_config(config: AuditConfig) -> None:
    positive = {
        "long_edge_ratio": config.long_edge_ratio,
        "short_edge_ratio": config.short_edge_ratio,
        "normal_jump_degrees": config.normal_jump_degrees,
        "condition_number": config.condition_number,
        "symmetric_stretch": config.symmetric_stretch,
        "symmetric_dirichlet": config.symmetric_dirichlet,
        "area_ratio_low": config.area_ratio_low,
        "area_ratio_high": config.area_ratio_high,
        "shear": config.shear,
        "normal_step_ratio": config.normal_step_ratio,
        "nonlocal_distance_ratio": config.nonlocal_distance_ratio,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if config.long_edge_ratio <= config.short_edge_ratio:
        raise ValueError("long_edge_ratio must exceed short_edge_ratio")
    if config.normal_jump_degrees >= 180:
        raise ValueError("normal_jump_degrees must be less than 180")
    if config.condition_number <= 1:
        raise ValueError("condition_number must exceed 1")
    if config.symmetric_stretch <= 1:
        raise ValueError("symmetric_stretch must exceed 1")
    if config.area_ratio_low >= 1 or config.area_ratio_high <= 1:
        raise ValueError("area ratio thresholds must straddle 1")
    if config.shear >= 1:
        raise ValueError("shear must be less than 1")
    if config.normal_step_min_component_cells < 1:
        raise ValueError("normal_step_min_component_cells must be at least 1")
    for name, value in (
        ("expected_spacing_x", config.expected_spacing_x),
        ("expected_spacing_y", config.expected_spacing_y),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number when supplied")
    if config.max_proximity_points <= 0 or config.max_proximity_pairs < 0:
        raise ValueError("proximity limits must be positive (pairs may be zero)")
    if config.nonlocal_uv_exclusion < 0:
        raise ValueError("nonlocal_uv_exclusion must be non-negative")


def _assign_edge_scores(
    review_score: np.ndarray,
    horizontal_ratio: np.ndarray,
    vertical_ratio: np.ndarray,
) -> None:
    """Assign absolute log2 spacing deviation to adjacent cells."""
    with np.errstate(divide="ignore", invalid="ignore"):
        horizontal_score = np.clip(np.abs(np.log2(horizontal_ratio)) / 2.0, 0.0, 1.0)
        vertical_score = np.clip(np.abs(np.log2(vertical_ratio)) / 2.0, 0.0, 1.0)
    horizontal_score = np.nan_to_num(horizontal_score, nan=0.0, posinf=1.0, neginf=1.0)
    vertical_score = np.nan_to_num(vertical_score, nan=0.0, posinf=1.0, neginf=1.0)

    if review_score.size == 0:
        return
    # Horizontal edge [r, c]-[r, c+1] borders cells above/below.
    review_score[:] = np.maximum(review_score, horizontal_score[:-1, :])
    review_score[:] = np.maximum(review_score, horizontal_score[1:, :])
    # Vertical edge [r, c]-[r+1, c] borders cells left/right.
    review_score[:] = np.maximum(review_score, vertical_score[:, :-1])
    review_score[:] = np.maximum(review_score, vertical_score[:, 1:])


def _project_edge_flags_to_cells(
    target: np.ndarray,
    horizontal_flags: np.ndarray,
    vertical_flags: np.ndarray,
) -> None:
    """Project flagged vertex-grid edges onto every incident face cell."""
    if target.size == 0:
        return
    target[:] |= horizontal_flags[:-1, :]
    target[:] |= horizontal_flags[1:, :]
    target[:] |= vertical_flags[:, :-1]
    target[:] |= vertical_flags[:, 1:]


def _project_cell_adjacency_flags(
    target: np.ndarray,
    horizontal_flags: np.ndarray,
    vertical_flags: np.ndarray,
) -> None:
    """Project flags between neighboring cells onto both endpoint cells."""
    if target.size == 0:
        return
    target[:, :-1] |= horizontal_flags
    target[:, 1:] |= horizontal_flags
    target[:-1, :] |= vertical_flags
    target[1:, :] |= vertical_flags


def _vertex_flags_to_cells(vertex_flags: np.ndarray) -> np.ndarray:
    """Return cells incident to at least one flagged vertex."""
    if vertex_flags.shape[0] < 2 or vertex_flags.shape[1] < 2:
        return np.zeros(
            (max(vertex_flags.shape[0] - 1, 0), max(vertex_flags.shape[1] - 1, 0)),
            dtype=bool,
        )
    return (
        vertex_flags[:-1, :-1]
        | vertex_flags[:-1, 1:]
        | vertex_flags[1:, :-1]
        | vertex_flags[1:, 1:]
    )


def _normal_jump_metrics(
    cell_normals: np.ndarray,
    valid_cells: np.ndarray,
    threshold_degrees: float,
    review_score: np.ndarray,
    review_cue_mask: np.ndarray,
) -> dict[str, Any]:
    threshold_cos = math.cos(math.radians(threshold_degrees))

    horizontal_valid = valid_cells[:, :-1] & valid_cells[:, 1:]
    vertical_valid = valid_cells[:-1, :] & valid_cells[1:, :]
    horizontal_dot = np.sum(cell_normals[:, :-1] * cell_normals[:, 1:], axis=-1)
    vertical_dot = np.sum(cell_normals[:-1, :] * cell_normals[1:, :], axis=-1)
    horizontal_dot = np.clip(horizontal_dot, -1.0, 1.0)
    vertical_dot = np.clip(vertical_dot, -1.0, 1.0)
    horizontal_angle = np.degrees(np.arccos(horizontal_dot))
    vertical_angle = np.degrees(np.arccos(vertical_dot))
    horizontal_angle[~horizontal_valid] = np.nan
    vertical_angle[~vertical_valid] = np.nan

    horizontal_flag = horizontal_valid & (horizontal_dot < threshold_cos)
    vertical_flag = vertical_valid & (vertical_dot < threshold_cos)
    horizontal_flip = horizontal_valid & (horizontal_dot < 0)
    vertical_flip = vertical_valid & (vertical_dot < 0)

    _project_cell_adjacency_flags(
        review_cue_mask,
        horizontal_flag,
        vertical_flag,
    )

    if review_score.size:
        horizontal_score = np.nan_to_num(
            np.clip(horizontal_angle / 90.0, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        vertical_score = np.nan_to_num(
            np.clip(vertical_angle / 90.0, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        review_score[:, :-1] = np.maximum(review_score[:, :-1], horizontal_score)
        review_score[:, 1:] = np.maximum(review_score[:, 1:], horizontal_score)
        review_score[:-1, :] = np.maximum(review_score[:-1, :], vertical_score)
        review_score[1:, :] = np.maximum(review_score[1:, :], vertical_score)

    jump_examples = [
        {**item, "direction": "horizontal"}
        for item in _top_locations(horizontal_angle, horizontal_flag, limit=25)
    ] + [
        {**item, "direction": "vertical"}
        for item in _top_locations(vertical_angle, vertical_flag, limit=25)
    ]
    jump_examples.sort(key=lambda item: (-float(item["value"]), item["direction"], item["rc"]))
    flip_examples = [
        {**item, "direction": "horizontal"}
        for item in _top_locations(horizontal_angle, horizontal_flip, limit=25)
    ] + [
        {**item, "direction": "vertical"}
        for item in _top_locations(vertical_angle, vertical_flip, limit=25)
    ]
    flip_examples.sort(key=lambda item: (-float(item["value"]), item["direction"], item["rc"]))

    return {
        "angle_degrees": _summary(
            np.concatenate([horizontal_angle.ravel(), vertical_angle.ravel()])
        ),
        "jumps_above_threshold": int(horizontal_flag.sum() + vertical_flag.sum()),
        "orientation_flips": int(horizontal_flip.sum() + vertical_flip.sum()),
        "jump_examples": jump_examples[:50],
        "orientation_flip_examples": flip_examples[:50],
    }


def _coherent_normal_step_metrics(
    horizontal_edges: np.ndarray,
    vertical_edges: np.ndarray,
    horizontal_valid: np.ndarray,
    vertical_valid: np.ndarray,
    cell_normals: np.ndarray,
    cell_valid_normals: np.ndarray,
    cell_valid: np.ndarray,
    horizontal_reference: float | None,
    vertical_reference: float | None,
    config: AuditConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    """Find spatially coherent edges with a large surface-normal component.

    A single curved or noisy edge is weak evidence.  This cue keeps only
    connected cell-space bands formed by multiple adjacent candidate edges.
    """
    height = horizontal_edges.shape[0]
    width = vertical_edges.shape[1]
    vertex_normal_sum = np.zeros((height, width, 3), dtype=np.float64)
    vertex_normal_count = np.zeros((height, width), dtype=np.int32)
    contribution = np.where(
        cell_valid_normals[..., None],
        cell_normals,
        0.0,
    )
    for row_slice, col_slice in (
        (slice(None, -1), slice(None, -1)),
        (slice(None, -1), slice(1, None)),
        (slice(1, None), slice(None, -1)),
        (slice(1, None), slice(1, None)),
    ):
        vertex_normal_sum[row_slice, col_slice] += contribution
        vertex_normal_count[row_slice, col_slice] += cell_valid_normals
    vertex_normals, vertex_normal_lengths = _safe_unit(vertex_normal_sum)
    vertex_normal_valid = (
        (vertex_normal_count > 0)
        & np.isfinite(vertex_normal_lengths)
        & (vertex_normal_lengths > 1e-12)
    )

    horizontal_normal_sum = vertex_normals[:, :-1] + vertex_normals[:, 1:]
    horizontal_normal, horizontal_normal_length = _safe_unit(horizontal_normal_sum)
    vertical_normal_sum = vertex_normals[:-1, :] + vertex_normals[1:, :]
    vertical_normal, vertical_normal_length = _safe_unit(vertical_normal_sum)
    horizontal_score = np.full(horizontal_valid.shape, np.nan, dtype=np.float64)
    vertical_score = np.full(vertical_valid.shape, np.nan, dtype=np.float64)
    horizontal_supported = (
        horizontal_valid
        & vertex_normal_valid[:, :-1]
        & vertex_normal_valid[:, 1:]
        & np.isfinite(horizontal_normal_length)
        & (horizontal_normal_length > 1e-12)
    )
    vertical_supported = (
        vertical_valid
        & vertex_normal_valid[:-1, :]
        & vertex_normal_valid[1:, :]
        & np.isfinite(vertical_normal_length)
        & (vertical_normal_length > 1e-12)
    )
    if horizontal_reference is not None:
        horizontal_score[horizontal_supported] = np.abs(
            np.sum(
                horizontal_edges[horizontal_supported]
                * horizontal_normal[horizontal_supported],
                axis=-1,
            )
        ) / horizontal_reference
    if vertical_reference is not None:
        vertical_score[vertical_supported] = np.abs(
            np.sum(
                vertical_edges[vertical_supported]
                * vertical_normal[vertical_supported],
                axis=-1,
            )
        ) / vertical_reference

    horizontal_candidates = (
        horizontal_supported
        & np.isfinite(horizontal_score)
        & (horizontal_score >= config.normal_step_ratio)
    )
    vertical_candidates = (
        vertical_supported
        & np.isfinite(vertical_score)
        & (vertical_score >= config.normal_step_ratio)
    )
    candidate_cells = np.zeros(cell_valid.shape, dtype=bool)
    _project_edge_flags_to_cells(
        candidate_cells,
        horizontal_candidates,
        vertical_candidates,
    )
    candidate_cells &= cell_valid
    labels, sizes = label_components_8(candidate_cells)
    retained_labels = {
        index
        for index, size in enumerate(sizes, start=1)
        if size >= config.normal_step_min_component_cells
    }
    coherent_cells = (
        np.isin(labels, list(retained_labels))
        if retained_labels
        else np.zeros(cell_valid.shape, dtype=bool)
    )
    component_sizes = sorted(
        (sizes[index - 1] for index in retained_labels),
        reverse=True,
    )
    examples = [
        {**item, "direction": "horizontal"}
        for item in _top_locations(
            horizontal_score,
            horizontal_candidates,
            limit=25,
        )
    ] + [
        {**item, "direction": "vertical"}
        for item in _top_locations(
            vertical_score,
            vertical_candidates,
            limit=25,
        )
    ]
    examples.sort(
        key=lambda item: (-float(item["value"]), item["direction"], item["rc"])
    )
    return (
        {
            "normal_component_ratio_threshold": config.normal_step_ratio,
            "minimum_component_cells": config.normal_step_min_component_cells,
            "horizontal_score": _summary(horizontal_score),
            "vertical_score": _summary(vertical_score),
            "candidate_edges": int(
                horizontal_candidates.sum() + vertical_candidates.sum()
            ),
            "coherent_components": len(component_sizes),
            "coherent_component_sizes": component_sizes,
            "coherent_cells": int(coherent_cells.sum()),
            "candidate_edge_examples": examples[:50],
        },
        coherent_cells,
    )


def _nonlocal_proximity(
    coordinates: np.ndarray,
    valid: np.ndarray,
    nominal_spacing: float | None,
    config: AuditConfig,
    cue_vertices: np.ndarray | None = None,
) -> dict[str, Any]:
    valid_count = int(valid.sum())
    if (
        valid_count == 0
        or nominal_spacing is None
        or not math.isfinite(nominal_spacing)
        or nominal_spacing <= 0
    ):
        return {
            "sampled_points": 0,
            "sample_stride": None,
            "sampled_points_capped": False,
            "distance_threshold": None,
            "pair_count": 0,
            "pairs_truncated": False,
            "pairs": [],
        }

    stride = max(1, int(math.ceil(math.sqrt(valid_count / config.max_proximity_points))))
    rows, cols = np.nonzero(valid[::stride, ::stride])
    rows = rows.astype(np.int64) * stride
    cols = cols.astype(np.int64) * stride
    sampled_points_capped = rows.size > config.max_proximity_points
    if sampled_points_capped:
        # A sparse mask can align with the regular stride and otherwise defeat
        # the configured point limit. Keep a deterministic, evenly spaced
        # subset of the strided valid points.
        selection = (
            np.arange(config.max_proximity_points, dtype=np.int64) * rows.size
        ) // config.max_proximity_points
        rows = rows[selection]
        cols = cols[selection]
    points = coordinates[rows, cols].astype(np.float64, copy=False)

    distance_threshold = config.nonlocal_distance_ratio * nominal_spacing
    if distance_threshold <= 0:
        raise ValueError("Nonlocal distance threshold must be positive")

    buckets: dict[tuple[int, int, int], list[int]] = {}
    pair_records: list[dict[str, Any]] = []
    total_pairs = 0
    truncated = False

    offsets = tuple(product((-1, 0, 1), repeat=3))
    for index, point in enumerate(points):
        bucket = tuple(np.floor(point / distance_threshold).astype(np.int64).tolist())
        for offset in offsets:
            neighbor_key = (
                bucket[0] + offset[0],
                bucket[1] + offset[1],
                bucket[2] + offset[2],
            )
            for previous in buckets.get(neighbor_key, ()):
                if (
                    max(abs(int(rows[index] - rows[previous])), abs(int(cols[index] - cols[previous])))
                    <= config.nonlocal_uv_exclusion
                ):
                    continue
                distance = float(np.linalg.norm(point - points[previous]))
                if distance >= distance_threshold:
                    continue
                total_pairs += 1
                if cue_vertices is not None:
                    cue_vertices[rows[previous], cols[previous]] = True
                    cue_vertices[rows[index], cols[index]] = True
                if len(pair_records) < config.max_proximity_pairs:
                    pair_records.append(
                        {
                            "a_rc": [int(rows[previous]), int(cols[previous])],
                            "b_rc": [int(rows[index]), int(cols[index])],
                            "distance": distance,
                        }
                    )
                else:
                    truncated = True
        buckets.setdefault(bucket, []).append(index)

    return {
        "sampled_points": int(points.shape[0]),
        "sample_stride": stride,
        "sampled_points_capped": sampled_points_capped,
        "distance_threshold": distance_threshold,
        "uv_exclusion": config.nonlocal_uv_exclusion,
        "pair_count": total_pairs,
        "pairs_truncated": truncated,
        "pairs": pair_records,
    }


def _findings(
    integrity: dict[str, Any],
    topology: dict[str, Any],
    geometry: dict[str, Any],
    proximity: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        findings.append({"level": level, "code": code, "message": message})

    if integrity["nonfinite_valid_vertices"]:
        add(
            "error",
            "nonfinite-valid",
            f"{integrity['nonfinite_valid_vertices']} vertices marked valid contain non-finite coordinates.",
        )
    if integrity["partial_sentinel_vertices"]:
        add(
            "error",
            "partial-sentinel",
            f"{integrity['partial_sentinel_vertices']} vertices use -1 in only some coordinate channels.",
        )
    if integrity["mask_coordinate_disagreements"]:
        add(
            "error",
            "mask-coordinate-disagreement",
            f"{integrity['mask_coordinate_disagreements']} explicit-mask entries disagree with z/finite validity.",
        )
    if topology["valid_vertex_components"] > 1:
        add(
            "review",
            "disconnected-vertices",
            f"Valid vertices form {topology['valid_vertex_components']} grid components.",
        )
    if topology["enclosed_invalid_regions"] > 0:
        add(
            "review",
            "enclosed-gaps",
            f"Found {topology['enclosed_invalid_regions']} enclosed invalid regions in the renderable face lattice.",
        )
    if geometry["long_edges"] > 0:
        add(
            "review",
            "long-edges",
            f"{geometry['long_edges']} grid edges exceed the configured observed-spacing ratio.",
        )
    if geometry["short_edges"] > 0:
        add(
            "review",
            "short-edges",
            f"{geometry['short_edges']} grid edges fall below the configured observed-spacing ratio.",
        )
    if geometry["degenerate_triangles"] > 0:
        add(
            "review",
            "degenerate-triangles",
            f"{geometry['degenerate_triangles']} official-split triangles are numerically degenerate.",
        )
    if geometry["folded_quads"] > 0:
        add(
            "review",
            "folded-quads",
            f"{geometry['folded_quads']} quads have opposed normals across Villa's official triangle split.",
        )
    if geometry["triangulation_sensitive_quads"] > 0:
        add(
            "review",
            "triangulation-sensitive",
            f"{geometry['triangulation_sensitive_quads']} quads change fold classification under the alternate diagonal.",
        )
    if geometry["normal_jumps"]["jumps_above_threshold"] > 0:
        add(
            "review",
            "normal-jumps",
            f"{geometry['normal_jumps']['jumps_above_threshold']} adjacent-cell normal jumps exceed the threshold.",
        )
    if geometry["coherent_normal_steps"]["coherent_components"] > 0:
        add(
            "review",
            "coherent-normal-step",
            (
                f"{geometry['coherent_normal_steps']['coherent_components']} "
                "spatially coherent band(s) contain a large surface-normal "
                "edge component."
            ),
        )
    if geometry["high_condition_cells"] > 0:
        add(
            "review",
            "anisotropic-cells",
            f"{geometry['high_condition_cells']} cells exceed the configured Jacobian condition number.",
        )
    if geometry["high_symmetric_stretch_cells"] > 0:
        add(
            "review",
            "symmetric-stretch",
            f"{geometry['high_symmetric_stretch_cells']} cells exceed the configured symmetric-stretch threshold.",
        )
    if geometry["high_area_distortion_cells"] > 0:
        add(
            "review",
            "area-distortion",
            f"{geometry['high_area_distortion_cells']} cells exceed the configured area-ratio range.",
        )
    if geometry["high_shear_cells"] > 0:
        add(
            "review",
            "high-shear",
            f"{geometry['high_shear_cells']} cells exceed the configured shear threshold.",
        )
    if geometry["high_symmetric_dirichlet_cells"] > 0:
        add(
            "review",
            "high-distortion",
            f"{geometry['high_symmetric_dirichlet_cells']} cells exceed the configured symmetric Dirichlet energy.",
        )
    isotropic_factor = geometry["target_spacing"]["isotropic_factor"]
    if isotropic_factor is not None and abs(float(isotropic_factor) - 1.0) > 0.15:
        add(
            "review",
            "global-spacing-drift",
            f"Observed geometric-mean spacing is {float(isotropic_factor):.3f}× the explicit target.",
        )
    anisotropy_factor = geometry["target_spacing"]["anisotropy_factor"]
    if anisotropy_factor is not None and float(anisotropy_factor) > 1.25:
        add(
            "review",
            "global-spacing-anisotropy",
            f"Directional target-relative spacing differs by {float(anisotropy_factor):.3f}×.",
        )
    if proximity["pair_count"] > 0:
        add(
            "review",
            "nonlocal-proximity",
            f"Found {proximity['pair_count']} sampled vertex pairs close in 3-D but distant in grid coordinates; these are contact candidates, not intersection proofs.",
        )

    level_order = {"error": 0, "review": 1, "info": 2}
    return sorted(findings, key=lambda item: (level_order.get(item["level"], 99), item["code"]))


def audit_mesh(data: TifxyzData, config: AuditConfig | None = None) -> dict[str, Any]:
    """Audit one TIFXYZ mesh and return JSON-serializable metrics plus a score map."""
    cfg = config or AuditConfig()
    _validate_config(cfg)
    coordinates = np.asarray(data.coordinates, dtype=np.float64)
    valid = np.asarray(data.valid, dtype=bool)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"Expected coordinates shaped (height, width, 3), got {coordinates.shape}")
    if valid.shape != coordinates.shape[:2]:
        raise ValueError(
            f"Validity mask shape {valid.shape} differs from coordinates {coordinates.shape[:2]}"
        )
    height, width = valid.shape

    finite_all = np.isfinite(coordinates).all(axis=-1)
    sentinel_channels = coordinates == -1.0
    sentinel_all = sentinel_channels.all(axis=-1)
    sentinel_any = sentinel_channels.any(axis=-1)
    derived_valid = (coordinates[..., 2] > 0) & np.isfinite(coordinates[..., 2])

    integrity = {
        "shape": [height, width],
        "vertex_count": int(height * width),
        "valid_vertex_count": int(valid.sum()),
        "valid_vertex_fraction": float(valid.mean()) if valid.size else 0.0,
        "nonfinite_vertices": int((~finite_all).sum()),
        "nonfinite_valid_vertices": int((valid & ~finite_all).sum()),
        "sentinel_vertices": int(sentinel_all.sum()),
        "partial_sentinel_vertices": int((sentinel_any & ~sentinel_all).sum()),
        "mask_coordinate_disagreements": (
            int((valid ^ derived_valid).sum()) if data.explicit_mask is not None else 0
        ),
    }

    cell_valid = valid_quad_mask(valid)
    _, vertex_component_sizes = label_components_4(valid)
    _, cell_component_sizes = label_components_4(cell_valid)
    hole_labels, hole_sizes = enclosed_invalid_regions(
        cell_valid,
        background_connectivity=8,
    )
    incident_vertices = np.zeros(valid.shape, dtype=bool)
    if cell_valid.size:
        incident_vertices[:-1, :-1] |= cell_valid
        incident_vertices[:-1, 1:] |= cell_valid
        incident_vertices[1:, :-1] |= cell_valid
        incident_vertices[1:, 1:] |= cell_valid
    topology = {
        "valid_vertex_components": len(vertex_component_sizes),
        "valid_vertex_component_sizes": sorted(vertex_component_sizes, reverse=True),
        "valid_quad_count": int(cell_valid.sum()),
        "valid_quad_components": len(cell_component_sizes),
        "valid_quad_component_sizes": sorted(cell_component_sizes, reverse=True),
        "enclosed_invalid_regions": len(hole_sizes),
        "enclosed_invalid_region_sizes": sorted(hole_sizes, reverse=True),
        "enclosed_invalid_region_details": _region_summaries(hole_labels),
        "valid_vertices_without_faces": int((valid & ~incident_vertices).sum()),
    }

    horizontal = coordinates[:, 1:] - coordinates[:, :-1]
    vertical = coordinates[1:, :] - coordinates[:-1, :]
    horizontal_length = np.linalg.norm(horizontal, axis=-1)
    vertical_length = np.linalg.norm(vertical, axis=-1)
    horizontal_valid = valid[:, 1:] & valid[:, :-1] & np.isfinite(horizontal_length)
    vertical_valid = valid[1:, :] & valid[:-1, :] & np.isfinite(vertical_length)
    horizontal_length[~horizontal_valid] = np.nan
    vertical_length[~vertical_valid] = np.nan

    observed_horizontal_spacing = _spacing_reference(None, horizontal_length)
    observed_vertical_spacing = _spacing_reference(None, vertical_length)
    horizontal_reference = _spacing_reference(
        cfg.expected_spacing_x,
        horizontal_length,
    )
    vertical_reference = _spacing_reference(
        cfg.expected_spacing_y,
        vertical_length,
    )

    horizontal_ratio = (
        horizontal_length / observed_horizontal_spacing
        if observed_horizontal_spacing is not None
        else np.full(horizontal_length.shape, np.nan, dtype=np.float64)
    )
    vertical_ratio = (
        vertical_length / observed_vertical_spacing
        if observed_vertical_spacing is not None
        else np.full(vertical_length.shape, np.nan, dtype=np.float64)
    )

    p00 = coordinates[:-1, :-1]
    p01 = coordinates[:-1, 1:]
    p10 = coordinates[1:, :-1]
    p11 = coordinates[1:, 1:]
    edge_u = p01 - p00
    edge_v = p10 - p00
    bilinear_twist = p11 - p10 - p01 + p00
    center_u = edge_u + 0.5 * bilinear_twist
    center_v = edge_v + 0.5 * bilinear_twist

    # Match vc_tifxyz2obj's p10-p01 diagonal.
    triangle_a = np.cross(p01 - p00, p10 - p00)
    triangle_b = np.cross(p11 - p01, p10 - p01)
    unit_a, double_area_a = _safe_unit(triangle_a)
    unit_b, double_area_b = _safe_unit(triangle_b)
    triangle_a_denominator = np.linalg.norm(p01 - p00, axis=-1) * np.linalg.norm(
        p10 - p00,
        axis=-1,
    )
    triangle_b_denominator = np.linalg.norm(p11 - p01, axis=-1) * np.linalg.norm(
        p10 - p01,
        axis=-1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        triangle_a_sine = double_area_a / triangle_a_denominator
        triangle_b_sine = double_area_b / triangle_b_denominator
    triangle_a_sine[~cell_valid] = np.nan
    triangle_b_sine[~cell_valid] = np.nan
    degenerate_a = cell_valid & (
        ~np.isfinite(triangle_a_sine) | (triangle_a_sine <= 1e-6)
    )
    degenerate_b = cell_valid & (
        ~np.isfinite(triangle_b_sine) | (triangle_b_sine <= 1e-6)
    )
    near_degenerate_a = cell_valid & np.isfinite(triangle_a_sine) & (triangle_a_sine <= 1e-3)
    near_degenerate_b = cell_valid & np.isfinite(triangle_b_sine) & (triangle_b_sine <= 1e-3)
    fold_cosine = np.sum(unit_a * unit_b, axis=-1)
    fold_cosine[~cell_valid] = np.nan
    folded = cell_valid & (fold_cosine < 0)

    alternative_a = np.cross(p01 - p00, p11 - p00)
    alternative_b = np.cross(p11 - p00, p10 - p00)
    alternative_unit_a, _ = _safe_unit(alternative_a)
    alternative_unit_b, _ = _safe_unit(alternative_b)
    alternative_fold_cosine = np.sum(alternative_unit_a * alternative_unit_b, axis=-1)
    alternative_fold_cosine[~cell_valid] = np.nan
    alternative_folded = cell_valid & (alternative_fold_cosine < 0)
    folded_both_diagonals = folded & alternative_folded
    triangulation_sensitive = folded ^ alternative_folded

    area = 0.5 * (double_area_a + double_area_b)
    area[~cell_valid] = np.nan
    cell_normals, cell_normal_length = _safe_unit(unit_a + unit_b)
    cell_valid_normals = cell_valid & np.isfinite(cell_normal_length) & (cell_normal_length > 1e-12)

    normalized_du = (
        center_u / horizontal_reference
        if horizontal_reference is not None
        else np.full(center_u.shape, np.nan, dtype=np.float64)
    )
    normalized_dv = (
        center_v / vertical_reference
        if vertical_reference is not None
        else np.full(center_v.shape, np.nan, dtype=np.float64)
    )
    gram_uu = np.sum(normalized_du * normalized_du, axis=-1)
    gram_vv = np.sum(normalized_dv * normalized_dv, axis=-1)
    gram_uv = np.sum(normalized_du * normalized_dv, axis=-1)
    trace = gram_uu + gram_vv
    determinant = gram_uu * gram_vv - gram_uv * gram_uv
    discriminant = np.maximum(trace * trace - 4.0 * determinant, 0.0)
    lambda_max = np.maximum(0.5 * (trace + np.sqrt(discriminant)), 0.0)
    lambda_min = np.maximum(0.5 * (trace - np.sqrt(discriminant)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_max = np.sqrt(lambda_max)
        sigma_min = np.sqrt(lambda_min)
        condition = sigma_max / sigma_min
        symmetric_stretch = np.maximum(sigma_max, 1.0 / sigma_min)
        area_ratio = sigma_max * sigma_min
        shear = np.abs(gram_uv) / np.sqrt(gram_uu * gram_vv)
        symmetric_dirichlet = lambda_max + lambda_min + 1.0 / lambda_max + 1.0 / lambda_min
    for metric in (
        condition,
        symmetric_stretch,
        area_ratio,
        shear,
        symmetric_dirichlet,
    ):
        metric[~cell_valid] = np.nan

    horizontal_long = horizontal_ratio > cfg.long_edge_ratio
    vertical_long = vertical_ratio > cfg.long_edge_ratio
    horizontal_short = horizontal_ratio < cfg.short_edge_ratio
    vertical_short = vertical_ratio < cfg.short_edge_ratio
    high_condition = cell_valid & (condition > cfg.condition_number)
    high_stretch = cell_valid & (symmetric_stretch > cfg.symmetric_stretch)
    high_area_distortion = cell_valid & (
        (area_ratio < cfg.area_ratio_low) | (area_ratio > cfg.area_ratio_high)
    )
    high_shear = cell_valid & (shear > cfg.shear)
    high_dirichlet = cell_valid & (symmetric_dirichlet > cfg.symmetric_dirichlet)

    review_cue_mask = np.zeros(cell_valid.shape, dtype=bool)
    _project_edge_flags_to_cells(
        review_cue_mask,
        horizontal_long | horizontal_short,
        vertical_long | vertical_short,
    )
    review_cue_mask |= (
        degenerate_a
        | degenerate_b
        | folded
        | triangulation_sensitive
        | high_condition
        | high_stretch
        | high_area_distortion
        | high_shear
        | high_dirichlet
    )
    review_cue_mask &= cell_valid

    review_score = np.zeros(cell_valid.shape, dtype=np.float64)
    if review_score.size:
        review_score[~cell_valid] = np.nan
        _assign_edge_scores(review_score, horizontal_ratio, vertical_ratio)
        condition_score = np.clip(
            np.log2(np.maximum(condition, 1.0)) / math.log2(max(cfg.condition_number, 1.0001)),
            0.0,
            1.0,
        )
        stretch_score = np.clip(
            np.log2(np.maximum(symmetric_stretch, 1.0))
            / math.log2(max(cfg.symmetric_stretch, 1.0001)),
            0.0,
            1.0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            area_deviation = np.maximum(area_ratio, 1.0 / area_ratio)
            area_threshold = max(cfg.area_ratio_high, 1.0 / cfg.area_ratio_low)
            area_score = np.clip(
                np.log2(np.maximum(area_deviation, 1.0)) / math.log2(area_threshold),
                0.0,
                1.0,
            )
        shear_score = np.clip(shear / cfg.shear, 0.0, 1.0)
        for score in (condition_score, stretch_score, area_score, shear_score):
            finite_score = np.nan_to_num(
                score,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            review_score[:] = np.maximum(review_score, finite_score)
        review_score[degenerate_a | degenerate_b] = 1.0
        review_score[folded] = 1.0

    normal_jumps = _normal_jump_metrics(
        cell_normals,
        cell_valid_normals,
        cfg.normal_jump_degrees,
        review_score,
        review_cue_mask,
    )
    coherent_normal_steps, coherent_normal_step_cells = (
        _coherent_normal_step_metrics(
            horizontal,
            vertical,
            horizontal_valid,
            vertical_valid,
            cell_normals,
            cell_valid_normals,
            cell_valid,
            horizontal_reference,
            vertical_reference,
            cfg,
        )
    )
    long_edges = int(horizontal_long.sum() + vertical_long.sum())
    short_edges = int(horizontal_short.sum() + vertical_short.sum())
    long_edge_examples = [
        {**item, "direction": "horizontal"}
        for item in _top_locations(
            horizontal_ratio,
            horizontal_long,
            limit=25,
        )
    ] + [
        {**item, "direction": "vertical"}
        for item in _top_locations(
            vertical_ratio,
            vertical_long,
            limit=25,
        )
    ]
    long_edge_examples.sort(
        key=lambda item: (-float(item["value"]), item["direction"], item["rc"])
    )
    short_edge_examples = [
        {**item, "direction": "horizontal"}
        for item in _top_locations(
            horizontal_ratio,
            horizontal_short,
            limit=25,
            largest=False,
        )
    ] + [
        {**item, "direction": "vertical"}
        for item in _top_locations(
            vertical_ratio,
            vertical_short,
            limit=25,
            largest=False,
        )
    ]
    short_edge_examples.sort(
        key=lambda item: (float(item["value"]), item["direction"], item["rc"])
    )
    degenerate_triangle_examples = [
        {**item, "triangle": "official_a"}
        for item in _top_locations(
            triangle_a_sine,
            degenerate_a,
            limit=25,
            largest=False,
        )
    ] + [
        {**item, "triangle": "official_b"}
        for item in _top_locations(
            triangle_b_sine,
            degenerate_b,
            limit=25,
            largest=False,
        )
    ]
    degenerate_triangle_examples.sort(
        key=lambda item: (
            0 if item["value"] is None else 1,
            float(item["value"]) if item["value"] is not None else 0.0,
            item["triangle"],
            item["rc"],
        )
    )
    target_spacing: dict[str, Any] = {
        "expected_x": cfg.expected_spacing_x,
        "expected_y": cfg.expected_spacing_y,
        "observed_to_expected_x": _ratio(
            observed_horizontal_spacing,
            cfg.expected_spacing_x,
        )
        if observed_horizontal_spacing is not None and cfg.expected_spacing_x is not None
        else None,
        "observed_to_expected_y": _ratio(
            observed_vertical_spacing,
            cfg.expected_spacing_y,
        )
        if observed_vertical_spacing is not None and cfg.expected_spacing_y is not None
        else None,
        "isotropic_factor": None,
        "anisotropy_factor": None,
    }
    if (
        target_spacing["observed_to_expected_x"] is not None
        and target_spacing["observed_to_expected_y"] is not None
    ):
        factor_x = float(target_spacing["observed_to_expected_x"])
        factor_y = float(target_spacing["observed_to_expected_y"])
        target_spacing["isotropic_factor"] = math.sqrt(factor_x * factor_y)
        target_spacing["anisotropy_factor"] = max(factor_x / factor_y, factor_y / factor_x)

    observed_grid_area = (
        int(cell_valid.sum()) * observed_horizontal_spacing * observed_vertical_spacing
        if observed_horizontal_spacing is not None and observed_vertical_spacing is not None
        else None
    )
    geometry = {
        "normalization_mode": (
            "target_relative"
            if cfg.expected_spacing_x is not None or cfg.expected_spacing_y is not None
            else "observed_regularized"
        ),
        "observed_spacing_x": observed_horizontal_spacing,
        "observed_spacing_y": observed_vertical_spacing,
        "normalization_spacing_x": horizontal_reference,
        "normalization_spacing_y": vertical_reference,
        "target_spacing": target_spacing,
        "horizontal_edge_length": _summary(horizontal_length),
        "vertical_edge_length": _summary(vertical_length),
        "horizontal_observed_ratio": _summary(horizontal_ratio),
        "vertical_observed_ratio": _summary(vertical_ratio),
        "long_edges": long_edges,
        "short_edges": short_edges,
        "long_edge_examples": long_edge_examples[:50],
        "short_edge_examples": short_edge_examples[:50],
        "surface_area_voxel2": float(np.nansum(area)),
        "observed_grid_area_voxel2": observed_grid_area,
        "surface_to_observed_grid_area_ratio": (
            _ratio(float(np.nansum(area)), observed_grid_area)
            if observed_grid_area is not None
            else None
        ),
        "quad_area": _summary(area),
        "triangle_normalized_area_sine": _summary(
            np.concatenate([triangle_a_sine.ravel(), triangle_b_sine.ravel()])
        ),
        "degenerate_triangles": int(degenerate_a.sum() + degenerate_b.sum()),
        "near_degenerate_triangles": int(
            near_degenerate_a.sum() + near_degenerate_b.sum()
        ),
        "degenerate_triangle_examples": degenerate_triangle_examples[:50],
        "official_diagonal_fold_cosine": _summary(fold_cosine),
        "alternate_diagonal_fold_cosine": _summary(alternative_fold_cosine),
        "folded_quads": int(folded.sum()),
        "folded_quads_both_diagonals": int(folded_both_diagonals.sum()),
        "triangulation_sensitive_quads": int(triangulation_sensitive.sum()),
        "folded_quad_examples": _top_locations(
            fold_cosine,
            folded,
            largest=False,
        ),
        "triangulation_sensitive_examples": _top_locations(
            np.minimum(fold_cosine, alternative_fold_cosine),
            triangulation_sensitive,
            largest=False,
        ),
        "condition_number": _summary(condition),
        "high_condition_cells": int(high_condition.sum()),
        "high_condition_examples": _top_locations(
            condition,
            high_condition,
        ),
        "symmetric_stretch": _summary(symmetric_stretch),
        "high_symmetric_stretch_cells": int(high_stretch.sum()),
        "high_symmetric_stretch_examples": _top_locations(
            symmetric_stretch,
            high_stretch,
        ),
        "area_ratio": _summary(area_ratio),
        "high_area_distortion_cells": int(high_area_distortion.sum()),
        "low_area_ratio_examples": _top_locations(
            area_ratio,
            high_area_distortion & (area_ratio < cfg.area_ratio_low),
            largest=False,
        ),
        "high_area_ratio_examples": _top_locations(
            area_ratio,
            high_area_distortion & (area_ratio > cfg.area_ratio_high),
        ),
        "shear": _summary(shear),
        "high_shear_cells": int(high_shear.sum()),
        "high_shear_examples": _top_locations(
            shear,
            high_shear,
        ),
        "symmetric_dirichlet": _summary(symmetric_dirichlet),
        "high_symmetric_dirichlet_cells": int(high_dirichlet.sum()),
        "high_symmetric_dirichlet_examples": _top_locations(
            symmetric_dirichlet,
            high_dirichlet,
        ),
        "normal_jumps": normal_jumps,
        "coherent_normal_steps": coherent_normal_steps,
    }

    valid_coordinates = coordinates[valid & finite_all]
    actual_bbox = None
    if valid_coordinates.size:
        actual_bbox = np.stack(
            [valid_coordinates.min(axis=0), valid_coordinates.max(axis=0)]
        )
    expected_bbox = _metadata_bbox(data.metadata)
    metadata_scale_x, metadata_scale_y = data.scale_xy
    parametric_cell_area = (
        1.0 / (metadata_scale_x * metadata_scale_y)
        if (
            metadata_scale_x is not None
            and metadata_scale_y is not None
            and math.isfinite(metadata_scale_x)
            and math.isfinite(metadata_scale_y)
            and metadata_scale_x > 0
            and metadata_scale_y > 0
        )
        else None
    )
    derived_parametric_area = (
        int(cell_valid.sum()) * parametric_cell_area
        if parametric_cell_area is not None
        else None
    )
    declared_area_raw = data.metadata.get("area_vx2", data.metadata.get("area"))
    metadata_checks: dict[str, Any] = {
        "actual_bbox": actual_bbox.tolist() if actual_bbox is not None else None,
        "declared_bbox": expected_bbox.tolist() if expected_bbox is not None else None,
        "bbox_max_abs_error": (
            float(np.max(np.abs(actual_bbox - expected_bbox)))
            if actual_bbox is not None and expected_bbox is not None
            else None
        ),
        "declared_scale_x": metadata_scale_x,
        "declared_scale_y": metadata_scale_y,
        "derived_parametric_cell_area_voxel2": parametric_cell_area,
        "derived_parametric_area_voxel2": derived_parametric_area,
        "declared_area_voxel2": declared_area_raw,
        "declared_vs_parametric_area_relative_error": None,
    }
    try:
        declared_area = float(declared_area_raw)
        metadata_checks["declared_vs_parametric_area_relative_error"] = _ratio(
            abs(derived_parametric_area - declared_area)
            if derived_parametric_area is not None
            else math.nan,
            declared_area,
        )
    except (TypeError, ValueError):
        pass

    proximity_references = (horizontal_reference, vertical_reference)
    nominal_for_proximity = (
        math.sqrt(horizontal_reference * vertical_reference)
        if (
            all(value is not None for value in proximity_references)
            and horizontal_reference is not None
            and vertical_reference is not None
        )
        else None
    )
    proximity_vertices = np.zeros(valid.shape, dtype=bool)
    proximity = _nonlocal_proximity(
        coordinates,
        valid & finite_all,
        nominal_for_proximity,
        cfg,
        cue_vertices=proximity_vertices,
    )
    proximity_cells = _vertex_flags_to_cells(proximity_vertices) & cell_valid
    review_cue_mask |= proximity_cells
    review_score[proximity_cells] = 1.0
    # Preserve the literal union produced by v0.1 before adding the new cue.
    # Subtracting the new cue from the final union would incorrectly erase
    # cells where old and new cue families overlap.
    v0_1_review_cue_mask = review_cue_mask.copy()
    review_cue_mask |= coherent_normal_step_cells
    review_score[coherent_normal_step_cells] = 1.0

    report = {
        "schema_version": "1.0.0",
        "tool": {"name": "tifxyz-doctor", "version": __version__},
        "source": {
            "path": str(data.path),
            "uuid": str(data.metadata.get("uuid", data.path.name)),
        },
        "config": asdict(cfg),
        "integrity": integrity,
        "topology": topology,
        "geometry": geometry,
        "metadata": metadata_checks,
        "nonlocal_proximity": proximity,
    }
    report["findings"] = _findings(integrity, topology, geometry, proximity)
    # Internal arrays are returned under a private key and removed by serializers.
    report["_arrays"] = {
        "review_score": review_score.astype(np.float32),
        "review_cue_mask": review_cue_mask,
        "v0_1_review_cue_mask": v0_1_review_cue_mask,
        "coherent_normal_step_cells": coherent_normal_step_cells,
        "valid_cells": cell_valid,
        "hole_labels": hole_labels,
    }
    return report


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without internal NumPy arrays."""
    return {key: value for key, value in report.items() if key != "_arrays"}
