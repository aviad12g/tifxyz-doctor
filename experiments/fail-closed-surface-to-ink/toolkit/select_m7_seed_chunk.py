#!/usr/bin/env python3
"""Select a fail-closed PHerc1447 m7 seed from one Zarr-v2 chunk.

The selector is deliberately local and deterministic.  It downloads exactly
one 192^3 binary m7 chunk, enumerates foreground voxels in a bounded cube, and
orders them by squared distance followed by a z/y/x tie break.  Each candidate
must look like one locally coherent sheet:

* a PCA normal can be estimated from nearby foreground voxels;
* the normal transect over offsets [-15, 15] contains exactly one foreground
  run, that run crosses offset zero, and its midpoint is near zero;
* four tangential probes can be snapped to nearby foreground voxels;
* every probe has a PCA normal coherent with every other normal; and
* every probe's own normal transect passes the same single-run test.

This is a seed-screening heuristic, not proof that a later grown patch remains
on one papyrus sheet.  It uses only NumPy plus VC3D's bundled Blosc library.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from sample_m7_seed_chunk import CHUNK_EDGE, DEFAULT_BLOSC, DEFAULT_PREFIX


@dataclass(frozen=True)
class SelectorConfig:
    search_radius: int = 32
    pca_radius: int = 5
    pca_min_points: int = 20
    maximum_normal_to_tangent_variance_ratio: float = 0.35
    minimum_tangent_variance_ratio: float = 0.05
    transect_half_length: int = 15
    maximum_run_center_offset: float = 2.0
    tangent_probe_distance: float = 7.0
    tangent_probe_snap_radius: float = 2.5
    minimum_normal_abs_dot: float = 0.94

    def validate(self) -> None:
        if self.search_radius < 0:
            raise ValueError("search_radius must be nonnegative")
        if self.pca_radius < 1:
            raise ValueError("pca_radius must be positive")
        if self.pca_min_points < 3:
            raise ValueError("pca_min_points must be at least three")
        if self.transect_half_length < 1:
            raise ValueError("transect_half_length must be positive")
        if self.maximum_run_center_offset < 0:
            raise ValueError("maximum_run_center_offset must be nonnegative")
        if self.tangent_probe_distance <= 0:
            raise ValueError("tangent_probe_distance must be positive")
        if self.tangent_probe_snap_radius < 0:
            raise ValueError("tangent_probe_snap_radius must be nonnegative")
        if not 0 <= self.maximum_normal_to_tangent_variance_ratio <= 1:
            raise ValueError(
                "maximum_normal_to_tangent_variance_ratio must be in [0, 1]"
            )
        if not 0 <= self.minimum_tangent_variance_ratio <= 1:
            raise ValueError("minimum_tangent_variance_ratio must be in [0, 1]")
        if not 0 <= self.minimum_normal_abs_dot <= 1:
            raise ValueError("minimum_normal_abs_dot must be in [0, 1]")


@dataclass(frozen=True)
class NormalEstimate:
    normal_zyx: tuple[float, float, float]
    eigenvalues_ascending: tuple[float, float, float]
    foreground_point_count: int
    normal_to_tangent_variance_ratio: float
    tangent_variance_ratio: float


@dataclass(frozen=True)
class TransectEvaluation:
    passed: bool
    reason: str
    runs_inclusive_offsets: tuple[tuple[int, int], ...]
    crossing_run_center_offset: float | None
    sampled_local_zyx: tuple[tuple[int, int, int], ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def contiguous_true_runs(line: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Return half-open runs in a one-dimensional Boolean array."""

    source = np.asarray(line, dtype=bool)
    if source.ndim != 1:
        raise ValueError("line must be one-dimensional")
    padded = np.pad(source.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def _canonical_vector_xyz(vector_xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Give an unoriented PCA vector a deterministic sign in x/y/z order."""

    vector = np.asarray(vector_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("vector must be finite and nonzero")
    vector = vector / norm
    # np.argmax supplies a stable x, then y, then z tie break.
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0:
        vector = -vector
    return vector


def estimate_pca_normal(
    foreground: NDArray[np.bool_],
    point_zyx: Sequence[int],
    config: SelectorConfig,
) -> tuple[NormalEstimate | None, str]:
    """Estimate the sheet normal from foreground coordinates in a local cube."""

    point = np.asarray(point_zyx, dtype=np.int64)
    if point.shape != (3,):
        raise ValueError("point_zyx must contain exactly three coordinates")
    shape = np.asarray(foreground.shape, dtype=np.int64)
    if np.any(point < 0) or np.any(point >= shape):
        return None, "point_out_of_bounds"
    if not bool(foreground[tuple(point)]):
        return None, "point_not_foreground"

    radius = int(config.pca_radius)
    lower = np.maximum(0, point - radius)
    upper = np.minimum(shape, point + radius + 1)
    patch = foreground[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    coordinates = np.argwhere(patch).astype(np.float64)
    if coordinates.shape[0] < config.pca_min_points:
        return None, "too_few_pca_points"
    coordinates += lower
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, coordinates.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues.astype(np.float64), 0.0)
    if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= 1e-12:
        return None, "degenerate_pca"
    normal_ratio = float(eigenvalues[0] / eigenvalues[1])
    tangent_ratio = float(eigenvalues[1] / max(eigenvalues[2], 1e-12))
    if normal_ratio > config.maximum_normal_to_tangent_variance_ratio:
        return None, "not_locally_planar"
    if tangent_ratio < config.minimum_tangent_variance_ratio:
        return None, "line_like_pca_support"

    normal_xyz = _canonical_vector_xyz(eigenvectors[:, 0][::-1])
    normal_zyx = normal_xyz[::-1]
    estimate = NormalEstimate(
        normal_zyx=tuple(float(value) for value in normal_zyx),
        eigenvalues_ascending=tuple(float(value) for value in eigenvalues),
        foreground_point_count=int(coordinates.shape[0]),
        normal_to_tangent_variance_ratio=normal_ratio,
        tangent_variance_ratio=tangent_ratio,
    )
    return estimate, "pass"


def evaluate_normal_transect(
    foreground: NDArray[np.bool_],
    point_zyx: Sequence[int],
    normal_zyx: Sequence[float],
    config: SelectorConfig,
) -> TransectEvaluation:
    """Require one foreground run containing offset zero along a PCA normal."""

    point = np.asarray(point_zyx, dtype=np.float64)
    normal = np.asarray(normal_zyx, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if point.shape != (3,) or normal.shape != (3,) or normal_norm <= 0:
        raise ValueError("point_zyx and normal_zyx must be valid 3-vectors")
    normal = normal / normal_norm
    half = int(config.transect_half_length)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    coordinates = point[None, :] + offsets[:, None] * normal[None, :]
    # Nearest-voxel sampling matches the binary prediction semantics.  Tiny
    # epsilon makes exact half values deterministic independent of banker's
    # rounding, without changing non-ties.
    sampled = np.floor(coordinates + 0.5 + 1e-12).astype(np.int64)
    shape = np.asarray(foreground.shape, dtype=np.int64)
    sampled_tuple = tuple(tuple(int(value) for value in row) for row in sampled)
    if np.any(sampled < 0) or np.any(sampled >= shape):
        return TransectEvaluation(
            passed=False,
            reason="transect_out_of_bounds",
            runs_inclusive_offsets=(),
            crossing_run_center_offset=None,
            sampled_local_zyx=sampled_tuple,
        )
    values = foreground[tuple(sampled.T)]
    runs = contiguous_true_runs(values)
    inclusive_offsets = tuple(
        (int(start - half), int(stop - 1 - half)) for start, stop in runs
    )
    center_index = half
    crossing = [run for run in runs if run[0] <= center_index < run[1]]
    if len(runs) != 1:
        reason = "competing_foreground_run" if crossing else "no_center_foreground_run"
        return TransectEvaluation(
            passed=False,
            reason=reason,
            runs_inclusive_offsets=inclusive_offsets,
            crossing_run_center_offset=None,
            sampled_local_zyx=sampled_tuple,
        )
    if len(crossing) != 1:
        return TransectEvaluation(
            passed=False,
            reason="no_center_foreground_run",
            runs_inclusive_offsets=inclusive_offsets,
            crossing_run_center_offset=None,
            sampled_local_zyx=sampled_tuple,
        )
    start, stop = crossing[0]
    run_center = ((start - half) + (stop - 1 - half)) / 2.0
    if abs(run_center) > config.maximum_run_center_offset:
        return TransectEvaluation(
            passed=False,
            reason="foreground_run_not_centered",
            runs_inclusive_offsets=inclusive_offsets,
            crossing_run_center_offset=float(run_center),
            sampled_local_zyx=sampled_tuple,
        )
    return TransectEvaluation(
        passed=True,
        reason="pass",
        runs_inclusive_offsets=inclusive_offsets,
        crossing_run_center_offset=float(run_center),
        sampled_local_zyx=sampled_tuple,
    )


def _tangent_basis_zyx(normal_zyx: Sequence[float]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normal_xyz = np.asarray(normal_zyx, dtype=np.float64)[::-1]
    normal_xyz /= np.linalg.norm(normal_xyz)
    reference_xyz = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(normal_xyz)))]
    tangent1_xyz = np.cross(normal_xyz, reference_xyz)
    tangent1_xyz = _canonical_vector_xyz(tangent1_xyz)
    tangent2_xyz = np.cross(normal_xyz, tangent1_xyz)
    tangent2_xyz = _canonical_vector_xyz(tangent2_xyz)
    return tangent1_xyz[::-1], tangent2_xyz[::-1]


def _nearest_foreground(
    foreground: NDArray[np.bool_],
    target_zyx: NDArray[np.float64],
    radius: float,
) -> tuple[int, int, int] | None:
    shape = np.asarray(foreground.shape, dtype=np.int64)
    integer_radius = int(math.ceil(radius))
    center = np.floor(target_zyx + 0.5 + 1e-12).astype(np.int64)
    lower = np.maximum(0, center - integer_radius)
    upper = np.minimum(shape, center + integer_radius + 1)
    patch = foreground[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    coordinates = np.argwhere(patch).astype(np.int64)
    if coordinates.size == 0:
        return None
    coordinates += lower
    deltas = coordinates.astype(np.float64) - target_zyx[None, :]
    squared = np.einsum("ij,ij->i", deltas, deltas)
    within = squared <= float(radius) ** 2 + 1e-12
    if not np.any(within):
        return None
    coordinates = coordinates[within]
    squared = squared[within]
    # Distance is primary; the remaining keys establish a stable z/y/x tie.
    order = np.lexsort(
        (coordinates[:, 2], coordinates[:, 1], coordinates[:, 0], squared)
    )
    return tuple(int(value) for value in coordinates[int(order[0])])


def evaluate_candidate(
    foreground: NDArray[np.bool_],
    point_zyx: Sequence[int],
    config: SelectorConfig,
) -> dict[str, Any]:
    """Evaluate one foreground candidate and return an auditable record."""

    config.validate()
    point = tuple(int(value) for value in point_zyx)
    record: dict[str, Any] = {"local_zyx": list(point), "pass": False}
    center_normal, reason = estimate_pca_normal(foreground, point, config)
    if center_normal is None:
        record["reason"] = reason
        return record
    record["center_normal"] = asdict(center_normal)
    center_transect = evaluate_normal_transect(
        foreground, point, center_normal.normal_zyx, config
    )
    record["center_transect"] = asdict(center_transect)
    if not center_transect.passed:
        record["reason"] = f"center_{center_transect.reason}"
        return record

    tangent1, tangent2 = _tangent_basis_zyx(center_normal.normal_zyx)
    signed_tangents = (
        ("t1_minus", -tangent1),
        ("t1_plus", tangent1),
        ("t2_minus", -tangent2),
        ("t2_plus", tangent2),
    )
    probe_records: list[dict[str, Any]] = []
    probe_normals = [np.asarray(center_normal.normal_zyx, dtype=np.float64)]
    point_array = np.asarray(point, dtype=np.float64)
    seen_points: set[tuple[int, int, int]] = {point}
    for label, tangent in signed_tangents:
        target = point_array + config.tangent_probe_distance * tangent
        probe = _nearest_foreground(
            foreground, target, config.tangent_probe_snap_radius
        )
        probe_record: dict[str, Any] = {
            "label": label,
            "target_local_zyx": target.tolist(),
            "local_zyx": list(probe) if probe is not None else None,
        }
        if probe is None:
            probe_record.update(pass_=False, reason="no_nearby_foreground")
            probe_record["pass"] = probe_record.pop("pass_")
            probe_records.append(probe_record)
            record["tangential_probes"] = probe_records
            record["reason"] = f"{label}_no_nearby_foreground"
            return record
        if probe in seen_points:
            probe_record.update(pass_=False, reason="duplicate_probe")
            probe_record["pass"] = probe_record.pop("pass_")
            probe_records.append(probe_record)
            record["tangential_probes"] = probe_records
            record["reason"] = f"{label}_duplicate_probe"
            return record
        seen_points.add(probe)
        probe_normal, probe_reason = estimate_pca_normal(foreground, probe, config)
        if probe_normal is None:
            probe_record.update(pass_=False, reason=probe_reason)
            probe_record["pass"] = probe_record.pop("pass_")
            probe_records.append(probe_record)
            record["tangential_probes"] = probe_records
            record["reason"] = f"{label}_{probe_reason}"
            return record
        probe_vector = np.asarray(probe_normal.normal_zyx, dtype=np.float64)
        coherence = [
            float(abs(np.dot(probe_vector, other))) for other in probe_normals
        ]
        minimum_coherence = min(coherence)
        probe_record["normal"] = asdict(probe_normal)
        probe_record["normal_abs_dots_to_prior"] = coherence
        if minimum_coherence + 1e-12 < config.minimum_normal_abs_dot:
            probe_record["pass"] = False
            probe_record["reason"] = "incoherent_normal"
            probe_records.append(probe_record)
            record["tangential_probes"] = probe_records
            record["reason"] = f"{label}_incoherent_normal"
            return record
        transect = evaluate_normal_transect(
            foreground, probe, probe_normal.normal_zyx, config
        )
        probe_record["transect"] = asdict(transect)
        if not transect.passed:
            probe_record["pass"] = False
            probe_record["reason"] = transect.reason
            probe_records.append(probe_record)
            record["tangential_probes"] = probe_records
            record["reason"] = f"{label}_{transect.reason}"
            return record
        probe_record["pass"] = True
        probe_record["reason"] = "pass"
        probe_records.append(probe_record)
        probe_normals.append(probe_vector)

    record["tangential_probes"] = probe_records
    record["minimum_pairwise_normal_abs_dot"] = min(
        abs(float(np.dot(left, right)))
        for index, left in enumerate(probe_normals)
        for right in probe_normals[index + 1 :]
    )
    record["pass"] = True
    record["reason"] = "pass"
    return record


def select_candidate(
    chunk: NDArray[np.uint8] | NDArray[np.bool_],
    requested_local_zyx: Sequence[int],
    config: SelectorConfig,
) -> dict[str, Any]:
    """Return the nearest candidate that passes all deterministic checks."""

    config.validate()
    if tuple(chunk.shape) != (CHUNK_EDGE,) * 3:
        raise ValueError(f"chunk must have shape {(CHUNK_EDGE,) * 3}")
    foreground = np.asarray(chunk) == 255 if chunk.dtype != bool else np.asarray(chunk)
    requested = np.asarray(requested_local_zyx, dtype=np.int64)
    if requested.shape != (3,):
        raise ValueError("requested_local_zyx must contain three coordinates")
    radius = int(config.search_radius)
    lower = requested - radius
    upper = requested + radius + 1
    if np.any(lower < 0) or np.any(upper > np.asarray(chunk.shape)):
        raise ValueError("bounded search crosses a chunk edge; fetch more chunks")
    cube = foreground[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    candidates = np.argwhere(cube).astype(np.int64) + lower
    if candidates.size == 0:
        return {
            "pass": False,
            "reason": "no_foreground_in_search_cube",
            "foreground_candidate_count": 0,
            "evaluated_candidate_count": 0,
            "rejection_reason_counts": {},
            "selected": None,
        }
    offsets = candidates - requested
    squared = np.einsum("ij,ij->i", offsets, offsets)
    order = np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0], squared))
    rejection_counts: dict[str, int] = {}
    first_rejections: list[dict[str, Any]] = []
    for ordinal, candidate_index in enumerate(order, start=1):
        candidate = candidates[int(candidate_index)]
        evaluation = evaluate_candidate(foreground, candidate, config)
        evaluation["distance_voxels"] = float(math.sqrt(float(squared[candidate_index])))
        evaluation["offset_zyx"] = [int(value) for value in candidate - requested]
        if evaluation["pass"]:
            return {
                "pass": True,
                "reason": "pass",
                "foreground_candidate_count": int(candidates.shape[0]),
                "evaluated_candidate_count": ordinal,
                "rejection_reason_counts": rejection_counts,
                "first_rejections": first_rejections,
                "selected": evaluation,
            }
        rejection_reason = str(evaluation["reason"])
        rejection_counts[rejection_reason] = rejection_counts.get(rejection_reason, 0) + 1
        if len(first_rejections) < 12:
            first_rejections.append(evaluation)
    return {
        "pass": False,
        "reason": "no_candidate_passed",
        "foreground_candidate_count": int(candidates.shape[0]),
        "evaluated_candidate_count": int(candidates.shape[0]),
        "rejection_reason_counts": rejection_counts,
        "first_rejections": first_rejections,
        "selected": None,
    }


def load_single_chunk(
    xyz: Sequence[int], prefix: str, blosc_path: Path
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Download and decode the one chunk containing ``xyz``."""

    xyz_array = np.asarray(xyz, dtype=np.int64)
    zyx = xyz_array[::-1]
    chunk_zyx = zyx // CHUNK_EDGE
    chunk_url = prefix.rstrip("/") + "/" + "/".join(map(str, chunk_zyx))
    with urllib.request.urlopen(chunk_url, timeout=60) as response:
        compressed = response.read()
    library = ctypes.CDLL(str(blosc_path.resolve()))
    library.blosc_decompress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.blosc_decompress.restype = ctypes.c_int
    expected_size = CHUNK_EDGE**3
    source = ctypes.create_string_buffer(compressed)
    destination = ctypes.create_string_buffer(expected_size)
    decompressed = int(library.blosc_decompress(source, destination, expected_size))
    if decompressed != expected_size:
        raise RuntimeError(
            f"expected {expected_size} decompressed bytes, got {decompressed}"
        )
    raw = destination.raw
    chunk = np.frombuffer(raw, dtype=np.uint8).reshape((CHUNK_EDGE,) * 3).copy()
    provenance = {
        "chunk_index_zyx": [int(value) for value in chunk_zyx],
        "chunk_url": chunk_url,
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "decompressed_bytes": decompressed,
        "decompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "codec_path": str(blosc_path.resolve()),
        "codec_sha256": hashlib.sha256(blosc_path.read_bytes()).hexdigest(),
    }
    return chunk, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--radius", type=int, default=32)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    xyz = np.asarray([args.x, args.y, args.z], dtype=np.int64)
    chunk, provenance = load_single_chunk(xyz, args.prefix, args.blosc)
    chunk_zyx = np.asarray(provenance["chunk_index_zyx"], dtype=np.int64)
    requested_local_zyx = xyz[::-1] - chunk_zyx * CHUNK_EDGE
    config = SelectorConfig(search_radius=int(args.radius))
    selection = select_candidate(chunk, requested_local_zyx, config)
    selected = selection.get("selected")
    if selected is not None:
        selected_local = np.asarray(selected["local_zyx"], dtype=np.int64)
        selected_global_zyx = chunk_zyx * CHUNK_EDGE + selected_local
        selected["global_zyx"] = [int(value) for value in selected_global_zyx]
        selected["global_xyz"] = [int(value) for value in selected_global_zyx[::-1]]
        selected["normal_xyz"] = list(
            reversed(selected["center_normal"]["normal_zyx"])
        )

    result = {
        "schema_version": 1,
        "array_axis_order": "zyx",
        "requested_xyz": [int(value) for value in xyz],
        "requested_zyx": [int(value) for value in xyz[::-1]],
        "requested_local_zyx": [int(value) for value in requested_local_zyx],
        "required_value": 255,
        "config": asdict(config),
        "chunk": provenance,
        "selection": selection,
        "limitations": [
            "This validates only one local binary m7 chunk, not the raw CT signal.",
            "PCA normals and nearest-voxel transects are heuristics, not sheet identity proof.",
            "A passing seed does not validate any surface grown from that seed.",
            "Tangential checks cover four probes at one radius, not the full future perimeter.",
        ],
    }
    rendered = json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if selection["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
