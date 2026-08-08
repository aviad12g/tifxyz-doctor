#!/usr/bin/env python3
"""Raw-ray form of the preregistered issue-191 fusion readout.

Geometry and sampling match Jinhojeong's published physical-pilot reader at
commit 3a46b63. This form retains unrounded counts and adds the explicitly
preregistered site-centre detection denominator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


REFERENCE_READER_SHA256 = "a533ee940712f1a47111705e4d8fff4990ccffb0afb103c7337a68bff244781c"
SPAN = 12.0
STEP = 0.5
K = int(2 * SPAN / STEP) + 1
OFFSETS = np.linspace(-SPAN, SPAN, K)
CENTER = K // 2
N_SAMPLE = 20_000
SAMPLING_SEED = 1218


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_reader(path: Path) -> None:
    if not path.is_file() or sha256_file(path) != REFERENCE_READER_SHA256:
        raise RuntimeError("published fusion reader SHA-256 mismatch")


def sample_rays(
    probability: np.ndarray,
    gt_surface: np.ndarray,
    turn_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float32)
    gt_surface = np.asarray(gt_surface, dtype=bool)
    turn_id = np.asarray(turn_id, dtype=np.int16)
    if probability.shape != gt_surface.shape or probability.shape != turn_id.shape:
        raise ValueError("probability, GT, and turn_id must share a ZYX shape")
    if probability.ndim != 3 or not np.isfinite(probability).all():
        raise ValueError("probability must be a finite 3-D array")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("probability lies outside [0,1]")
    if np.any(turn_id < 0) or np.any((turn_id > 0) & ~gt_surface):
        raise ValueError("turn_id must be a nonnegative subset of GT surface")

    mask = turn_id > 0
    signed = (
        ndi.distance_transform_edt(~mask).astype(np.float32)
        - ndi.distance_transform_edt(mask).astype(np.float32)
    )
    smoothed = ndi.gaussian_filter(signed, 1.0)
    gradient = np.stack(np.gradient(smoothed), axis=0).astype(np.float32)
    surface = mask & ~ndi.binary_erosion(mask)
    points = np.argwhere(surface)
    if not len(points):
        raise ValueError("turn_id has no surface sites")
    rng = np.random.default_rng(SAMPLING_SEED)
    points = points[
        rng.choice(len(points), size=min(N_SAMPLE, len(points)), replace=False)
    ]
    normals = gradient[:, points[:, 0], points[:, 1], points[:, 2]].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6
    coordinates = (
        points[:, None, :].astype(np.float32)
        + OFFSETS[None, :, None] * normals[:, None, :]
    )
    flat = coordinates.reshape(-1, 3).T
    ray_turn = ndi.map_coordinates(
        turn_id.astype(np.float32), flat, order=0, mode="constant"
    ).reshape(len(points), K).astype(np.int16)
    ray_probability = ndi.map_coordinates(
        probability, flat, order=1, mode="constant"
    ).reshape(len(points), K).astype(np.float16)
    return ray_probability, ray_turn


def score_rays(
    ray_probability: np.ndarray, ray_turn: np.ndarray, threshold: float
) -> dict[str, int | float]:
    probability = np.asarray(ray_probability, dtype=np.float32)
    turn = np.asarray(ray_turn, dtype=np.int16)
    if probability.shape != turn.shape or probability.ndim != 2:
        raise ValueError("ray arrays must share an NxK shape")
    if probability.shape[1] != K:
        raise ValueError(f"ray width must be {K}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0,1]")

    neighbour_sites = 0
    detected_neighbour_sites = 0
    fused_detected_sites = 0
    control_sites = 0
    false_split_sites = 0
    for index in range(len(turn)):
        own = turn[index, CENTER]
        if own <= 0:
            continue
        row = turn[index]
        other = (row > 0) & (row != own)
        if other.any():
            nearest = np.flatnonzero(other)
            neighbour = nearest[np.argmin(np.abs(nearest - CENTER))]
            left, right = (
                (CENTER, neighbour) if neighbour > CENTER else (neighbour, CENTER)
            )
            neighbour_sites += 1
            detected = probability[index, CENTER] >= threshold
            detected_neighbour_sites += int(detected)
            fused_detected_sites += int(
                detected and (probability[index, left : right + 1] >= threshold).all()
            )
        else:
            left = CENTER
            while left > 0 and row[left - 1] == own:
                left -= 1
            right = CENTER
            while right < K - 1 and row[right + 1] == own:
                right += 1
            control_sites += 1
            false_split_sites += int(
                not (probability[index, left : right + 1] >= threshold).all()
            )

    return {
        "neighbour_sites": neighbour_sites,
        "detected_neighbour_sites": detected_neighbour_sites,
        "fused_detected_sites": fused_detected_sites,
        "control_sites": control_sites,
        "false_split_sites": false_split_sites,
        "site_center_detection_rate": detected_neighbour_sites / max(neighbour_sites, 1),
        "conditional_fusion_rate": fused_detected_sites
        / max(detected_neighbour_sites, 1),
        "false_split_rate": false_split_sites / max(control_sites, 1),
    }
