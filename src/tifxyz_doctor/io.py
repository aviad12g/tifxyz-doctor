"""TIFXYZ loading with explicit contract checks.

The Vesuvius Python reader uses ``z > 0`` when ``mask.tif`` is absent.  This
module follows that convention for the effective validity mask while retaining
enough information to report inconsistent sentinels, non-finite values, and a
mask that disagrees with the coordinate arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TifxyzData:
    """In-memory representation of one stored-resolution TIFXYZ surface."""

    path: Path
    coordinates: np.ndarray
    valid: np.ndarray
    metadata: dict[str, Any]
    explicit_mask: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.coordinates.shape[0]), int(self.coordinates.shape[1])

    @property
    def scale_xy(self) -> tuple[float | None, float | None]:
        """Return metadata scale as ``(x_scale, y_scale)``."""
        raw = self.metadata.get("scale")
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None, None
        try:
            sx, sy = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None, None
        return sx, sy


def _read_tiff(path: Path) -> np.ndarray:
    """Read a 2-D TIFF, preferring tifffile and falling back to Pillow."""
    tifffile_error: Exception | None = None
    try:
        import tifffile  # type: ignore

        data = tifffile.imread(path)
    except Exception as exc:
        # A common lightweight environment has tifffile but not imagecodecs;
        # official LZW/BigTIFF examples then fail there even though Pillow can
        # decode them. Preserve the first failure if both readers fail.
        tifffile_error = exc
        from PIL import Image

        try:
            with Image.open(path) as image:
                data = np.asarray(image)
        except Exception as pillow_error:
            detail = (
                f"tifffile={type(tifffile_error).__name__}: {tifffile_error}; "
                f"Pillow={type(pillow_error).__name__}: {pillow_error}"
            )
            raise ValueError(f"{path.name}: no TIFF reader succeeded ({detail})") from pillow_error

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"{path.name}: expected one 2-D image, got shape {array.shape}")
    return array


def load_tifxyz(path: str | Path) -> TifxyzData:
    """Load a TIFXYZ directory without silently repairing invalid input."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"TIFXYZ directory not found: {directory}")

    required = ("x.tif", "y.tif", "z.tif", "meta.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required TIFXYZ files: {', '.join(missing)}")

    with (directory / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("meta.json must contain a JSON object")

    components = [_read_tiff(directory / f"{axis}.tif") for axis in ("x", "y", "z")]
    shapes = {tuple(array.shape) for array in components}
    if len(shapes) != 1:
        detail = ", ".join(
            f"{axis}={array.shape}" for axis, array in zip(("x", "y", "z"), components)
        )
        raise ValueError(f"Coordinate array shapes differ: {detail}")

    coordinates = np.stack(
        [array.astype(np.float32, copy=False) for array in components],
        axis=-1,
    )

    mask_path = directory / "mask.tif"
    explicit_mask: np.ndarray | None = None
    if mask_path.is_file():
        candidate_mask = _read_tiff(mask_path) != 0
        # Match the current Villa Python reader: a differently sized mask is
        # ignored and validity falls back to finite z > 0. The independent raw
        # contract audit retains and reports the interoperability evidence.
        if candidate_mask.shape == coordinates.shape[:2]:
            explicit_mask = candidate_mask

    if explicit_mask is None:
        # Matches ScrollPrize/villa's TifxyzReader contract.
        valid = (coordinates[..., 2] > 0) & np.isfinite(coordinates[..., 2])
    else:
        valid = explicit_mask.copy()

    return TifxyzData(
        path=directory,
        coordinates=coordinates,
        valid=valid,
        metadata=metadata,
        explicit_mask=explicit_mask,
    )
