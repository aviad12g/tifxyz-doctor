#!/usr/bin/env python3
"""Fail-closed non-adjacent self-intersection audit for a TIFFXYZ quad mesh.

The stored grid is split along the same anti-diagonal used by the native
geometry validator. Triangle pairs that share a source-grid vertex are local
adjacency and are reported separately. All other pairs are screened with a
rigorous bounding-sphere superset and tested for 3-D triangle contact. The
minimum non-adjacent triangle distance is then used with a bilinear-deviation
bound to certify (or refuse to certify) a finer endpoint-aligned raster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from tifxyz_render_pipeline import load_tifxyz_asset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def triangulate_grid(points_xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quads = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    rows, columns = np.where(quads)
    width = valid.shape[1]
    v00 = rows * width + columns
    v01 = v00 + 1
    v10 = v00 + width
    v11 = v10 + 1
    indices = np.empty((2 * len(rows), 3), dtype=np.int64)
    indices[0::2] = np.stack((v00, v01, v10), axis=1)
    indices[1::2] = np.stack((v11, v10, v01), axis=1)
    return points_xyz.reshape(-1, 3)[indices], indices


def _orient2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tol: float) -> bool:
    o1, o2 = _orient2(a, b, c), _orient2(a, b, d)
    o3, o4 = _orient2(c, d, a), _orient2(c, d, b)
    if max(abs(o1), abs(o2), abs(o3), abs(o4)) <= tol:
        return bool(
            np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                   <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + tol)
        )
    return (o1 <= tol and o2 >= -tol or o2 <= tol and o1 >= -tol) and (
        o3 <= tol and o4 >= -tol or o4 <= tol and o3 >= -tol
    )


def _point_in_triangle_2d(p: np.ndarray, tri: np.ndarray, tol: float) -> bool:
    values = [_orient2(tri[k], tri[(k + 1) % 3], p) for k in range(3)]
    return not (min(values) < -tol and max(values) > tol)


def _coplanar_triangles_intersect(a: np.ndarray, b: np.ndarray, normal: np.ndarray, tol: float) -> bool:
    drop = int(np.argmax(np.abs(normal)))
    aa, bb = np.delete(a, drop, axis=1), np.delete(b, drop, axis=1)
    for i in range(3):
        for j in range(3):
            if _segments_intersect_2d(aa[i], aa[(i + 1) % 3], bb[j], bb[(j + 1) % 3], tol):
                return True
    return _point_in_triangle_2d(aa[0], bb, tol) or _point_in_triangle_2d(bb[0], aa, tol)


def _segment_triangle_intersect(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray, tol: float) -> bool:
    edge1, edge2 = tri[1] - tri[0], tri[2] - tri[0]
    direction = p1 - p0
    h = np.cross(direction, edge2)
    det = float(np.dot(edge1, h))
    if abs(det) <= tol:
        return False
    inv = 1.0 / det
    s = p0 - tri[0]
    u = inv * float(np.dot(s, h))
    if u < -tol or u > 1.0 + tol:
        return False
    q = np.cross(s, edge1)
    v = inv * float(np.dot(direction, q))
    if v < -tol or u + v > 1.0 + tol:
        return False
    t = inv * float(np.dot(edge2, q))
    return -tol <= t <= 1.0 + tol


def triangles_intersect(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> bool:
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    na_norm, nb_norm = float(np.linalg.norm(na)), float(np.linalg.norm(nb))
    if na_norm <= tol or nb_norm <= tol:
        raise ValueError("degenerate triangle supplied to intersection test")
    da = (b - a[0]) @ na
    db = (a - b[0]) @ nb
    scaled = tol * max(1.0, na_norm, nb_norm)
    if np.all(da > scaled) or np.all(da < -scaled) or np.all(db > scaled) or np.all(db < -scaled):
        return False
    cross_norm = float(np.linalg.norm(np.cross(na, nb)))
    if cross_norm <= tol * na_norm * nb_norm and max(np.max(np.abs(da)), np.max(np.abs(db))) <= scaled:
        return _coplanar_triangles_intersect(a, b, na, tol)
    for i in range(3):
        if _segment_triangle_intersect(a[i], a[(i + 1) % 3], b, tol):
            return True
        if _segment_triangle_intersect(b[i], b[(i + 1) % 3], a, tol):
            return True
    return False


def _point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> float:
    # Ericson, Real-Time Collision Detection, closest point on triangle.
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(ap))
    bp = p - b; d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(bp))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return float(np.linalg.norm(p - (a + (d1 / (d1 - d3)) * ab)))
    cp = p - c; d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(cp))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return float(np.linalg.norm(p - (a + (d2 / (d2 - d6)) * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return float(np.linalg.norm(p - (b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b))))
    denom = 1.0 / (va + vb + vc)
    return float(np.linalg.norm(p - (a + ab * vb * denom + ac * vc * denom)))


def _segment_segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = float(np.dot(d1, d1)), float(np.dot(d2, d2))
    eps = 1e-15
    if a <= eps and e <= eps:
        return float(np.linalg.norm(r))
    if a <= eps:
        s, t = 0.0, np.clip(float(np.dot(d2, r)) / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b, f = float(np.dot(d1, d2)), float(np.dot(d2, r))
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if abs(denom) > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def triangle_distance(a: np.ndarray, b: np.ndarray) -> float:
    if triangles_intersect(a, b):
        return 0.0
    values = [_point_triangle_distance(p, b) for p in a]
    values += [_point_triangle_distance(p, a) for p in b]
    for i in range(3):
        for j in range(3):
            values.append(_segment_segment_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]))
    return min(values)


def share_vertex(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.intersect1d(a, b, assume_unique=False).size)


def audit(surface: Path, *, voxel_um: float, fine_step: float) -> dict[str, Any]:
    asset = load_tifxyz_asset(surface, resolution="stored")
    points = asset.surface.points_xyz.astype(np.float64)
    triangles, indices = triangulate_grid(points, asset.surface.valid)
    centroids = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centroids[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(centroids)
    broad = sorted(tree.query_pairs(float(2.0 * radii.max() + 1e-9)))
    aabb_min, aabb_max = triangles.min(axis=1), triangles.max(axis=1)
    overlap_pairs: list[tuple[int, int]] = []
    intersections: list[tuple[int, int]] = []
    adjacent = 0
    for i, j in broad:
        if share_vertex(indices[i], indices[j]):
            adjacent += 1
            continue
        if np.all(aabb_max[i] >= aabb_min[j]) and np.all(aabb_max[j] >= aabb_min[i]):
            overlap_pairs.append((i, j))
            if triangles_intersect(triangles[i], triangles[j]):
                intersections.append((i, j))

    # Find an initial finite upper bound from centroid-nearest non-adjacent pairs.
    best = float("inf"); best_pair: tuple[int, int] | None = None
    _dist, nearest = tree.query(centroids, k=min(24, len(triangles)))
    for i, row in enumerate(np.atleast_2d(nearest)):
        for j in row[1:]:
            j = int(j)
            if i < j and not share_vertex(indices[i], indices[j]):
                value = triangle_distance(triangles[i], triangles[j])
                if value < best:
                    best, best_pair = value, (i, j)
    if not np.isfinite(best):
        raise RuntimeError("could not initialize a non-adjacent distance bound")
    # Any pair that can beat best must have centroid separation <= ri+rj+best.
    distance_candidates = sorted(tree.query_pairs(float(2.0 * radii.max() + best)))
    evaluated = 0
    for i, j in distance_candidates:
        if share_vertex(indices[i], indices[j]):
            continue
        center_distance = float(np.linalg.norm(centroids[i] - centroids[j]))
        if center_distance - radii[i] - radii[j] > best:
            continue
        value = triangle_distance(triangles[i], triangles[j])
        evaluated += 1
        if value < best:
            best, best_pair = value, (i, j)

    valid = asset.surface.valid
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    p00, p01 = points[:-1, :-1], points[:-1, 1:]
    p10, p11 = points[1:, :-1], points[1:, 1:]
    cross_term = p00 - p01 - p10 + p11
    coarse_bilinear_deviation = float(np.max(np.linalg.norm(cross_term[q], axis=1)) / 4.0)
    per_side_bound = coarse_bilinear_deviation * (1.0 + fine_step * fine_step)
    full_clearance_bound = best - 2.0 * per_side_bound
    passed_stored = len(intersections) == 0
    passed_full_bound = passed_stored and full_clearance_bound > 0.0
    return {
        "schema_version": 1,
        "status": "complete",
        "surface": {"path": str(surface.resolve()), "input_sha256": asset.input_sha256},
        "voxel_um": voxel_um,
        "triangle_split": "anti-diagonal: (00,01,10) and (11,10,01)",
        "triangle_count": int(len(triangles)),
        "shared_source_vertex_pairs_excluded": int(adjacent),
        "broad_phase_pair_count": int(len(broad)),
        "nonadjacent_aabb_overlap_pair_count": int(len(overlap_pairs)),
        "nonadjacent_intersection_count": int(len(intersections)),
        "nonadjacent_intersection_pairs": [list(pair) for pair in intersections[:100]],
        "minimum_nonadjacent_triangle_distance_voxels": best,
        "minimum_nonadjacent_triangle_distance_um": best * voxel_um,
        "minimum_distance_triangle_pair": list(best_pair) if best_pair else None,
        "distance_candidate_pair_count": int(len(distance_candidates)),
        "distance_exact_evaluation_count": int(evaluated),
        "fine_resampling_step_stored_pixels": fine_step,
        "maximum_bilinear_vs_stored_triangle_deviation_voxels": coarse_bilinear_deviation,
        "per_surface_full_res_deviation_bound_voxels": per_side_bound,
        "certified_full_res_nonadjacent_clearance_lower_bound_voxels": full_clearance_bound,
        "certified_full_res_nonadjacent_clearance_lower_bound_um": full_clearance_bound * voxel_um,
        "stored_nonadjacent_self_intersection_pass": passed_stored,
        "full_res_nonadjacent_clearance_bound_pass": passed_full_bound,
        "pass_to_raw_diagnostic": passed_full_bound,
        "limitations": [
            "Pairs sharing a source-grid vertex are local adjacency and excluded from the non-adjacent claim.",
            "The full-resolution claim is a conservative displacement/clearance bound, not an all-pairs fine-mesh intersection enumeration.",
            "Local full-raster folds, degeneracies, and manifold topology must be validated independently.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface", type=Path)
    parser.add_argument("--voxel-um", type=float, required=True)
    parser.add_argument("--fine-step", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit(args.surface, voxel_um=float(args.voxel_um), fine_step=float(args.fine_step))
    _atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "pass": result["pass_to_raw_diagnostic"], "intersections": result["nonadjacent_intersection_count"], "full_clearance_voxels": result["certified_full_res_nonadjacent_clearance_lower_bound_voxels"]}, sort_keys=True))
    return 0 if result["pass_to_raw_diagnostic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
