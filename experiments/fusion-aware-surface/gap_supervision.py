"""Exact inter-sheet gap supervision derived from instance-labelled sheets."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def disk(radius: int) -> np.ndarray:
    """Return a 2-D Euclidean disk with an integer voxel radius."""
    if radius < 1:
        raise ValueError("radius must be at least one voxel")
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def inter_sheet_gap_mask(
    turn_id: np.ndarray,
    surface: np.ndarray | None = None,
    *,
    radius: int = 3,
) -> np.ndarray:
    """Mark background voxels lying between two distinct sheet instances.

    `turn_id` is the painter's 2-D instance map (zero is air). Each instance is
    dilated by the same Euclidean disk. A background voxel is supervised as a
    gap only when the dilated support of at least two different instances
    reaches it. This excludes exterior background near a single sheet.

    `surface` may include join-only material outside `turn_id`; those voxels
    are always removed from the gap mask.
    """
    labels = np.asarray(turn_id)
    if labels.ndim != 2:
        raise ValueError(f"turn_id must be 2-D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("turn_id must have an integer dtype")
    if np.any(labels < 0):
        raise ValueError("turn_id cannot contain negative labels")

    material = labels > 0
    if surface is None:
        occupied = material
    else:
        occupied = np.asarray(surface) > 0
        if occupied.shape != labels.shape:
            raise ValueError(
                f"surface shape {occupied.shape} does not match {labels.shape}"
            )
        if np.any(material & ~occupied):
            raise ValueError("every positive turn_id voxel must be surface")

    support_count = np.zeros(labels.shape, dtype=np.uint16)
    footprint = disk(radius)
    for instance in np.unique(labels[material]):
        support_count += ndi.binary_dilation(
            labels == instance, structure=footprint
        ).astype(np.uint16)

    return (support_count >= 2) & ~occupied


def broadcast_gap_mask(mask_2d: np.ndarray, depth: int) -> np.ndarray:
    """Broadcast a 2-D gap mask into a writable ZYX boolean volume."""
    mask = np.asarray(mask_2d, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if depth < 1:
        raise ValueError("depth must be positive")
    return np.broadcast_to(mask, (depth,) + mask.shape).copy()

