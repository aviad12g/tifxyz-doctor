"""Grid-topology helpers with no SciPy dependency."""

from __future__ import annotations

from collections import deque

import numpy as np


_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _label_components(
    mask: np.ndarray,
    neighbors: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, list[int]]:
    """Label connected components using an explicit neighborhood."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {source.shape}")

    height, width = source.shape
    labels = np.zeros(source.shape, dtype=np.int32)
    sizes: list[int] = []
    component = 0

    for start_flat in np.flatnonzero(source & (labels == 0)):
        row, col = divmod(int(start_flat), width)
        if labels[row, col] != 0:
            continue
        component += 1
        labels[row, col] = component
        queue: deque[tuple[int, int]] = deque([(row, col)])
        size = 0
        while queue:
            current_row, current_col = queue.popleft()
            size += 1
            for delta_row, delta_col in neighbors:
                next_row = current_row + delta_row
                next_col = current_col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and source[next_row, next_col]
                    and labels[next_row, next_col] == 0
                ):
                    labels[next_row, next_col] = component
                    queue.append((next_row, next_col))
        sizes.append(size)

    return labels, sizes


def label_components_4(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Label 4-connected components; zero is background."""
    return _label_components(mask, _NEIGHBORS_4)


def label_components_8(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Label 8-connected components; zero is background."""
    return _label_components(mask, _NEIGHBORS_8)


def enclosed_invalid_regions(
    valid: np.ndarray,
    *,
    background_connectivity: int = 8,
) -> tuple[np.ndarray, list[int]]:
    """Return invalid regions that are not connected to the grid boundary.

    Face topology normally pairs 4-connected foreground with 8-connected
    background, avoiding ambiguous diagonal pinches.
    """
    valid_array = np.asarray(valid, dtype=bool)
    if valid_array.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {valid_array.shape}")
    if background_connectivity not in (4, 8):
        raise ValueError("background_connectivity must be 4 or 8")
    invalid = ~valid_array
    if not invalid.any():
        return np.zeros(valid_array.shape, dtype=np.int32), []
    if valid_array.size == 0:
        return np.zeros(valid_array.shape, dtype=np.int32), []

    height, width = valid_array.shape
    exterior = np.zeros(valid_array.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    boundary = np.zeros(valid_array.shape, dtype=bool)
    if height:
        boundary[0, :] = True
        boundary[-1, :] = True
    if width:
        boundary[:, 0] = True
        boundary[:, -1] = True
    for flat in np.flatnonzero(boundary & invalid):
        row, col = divmod(int(flat), width)
        exterior[row, col] = True
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        neighbors = _NEIGHBORS_8 if background_connectivity == 8 else _NEIGHBORS_4
        for delta_row, delta_col in neighbors:
            next_row = row + delta_row
            next_col = col + delta_col
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and invalid[next_row, next_col]
                and not exterior[next_row, next_col]
            ):
                exterior[next_row, next_col] = True
                queue.append((next_row, next_col))

    holes = invalid & ~exterior
    return (
        label_components_8(holes)
        if background_connectivity == 8
        else label_components_4(holes)
    )


def valid_quad_mask(valid_vertices: np.ndarray) -> np.ndarray:
    """Return cells whose four corner vertices are valid."""
    valid = np.asarray(valid_vertices, dtype=bool)
    if valid.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {valid.shape}")
    if min(valid.shape) < 2:
        return np.zeros((max(valid.shape[0] - 1, 0), max(valid.shape[1] - 1, 0)), dtype=bool)
    return valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
