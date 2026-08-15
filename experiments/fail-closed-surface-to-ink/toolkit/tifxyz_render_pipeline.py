"""End-to-end tiled rendering support for native Vesuvius volumes.

This module is the I/O and orchestration layer around
``native_surface_sampler``.  It deliberately keeps the coordinate conventions
visible at every boundary:

* source volumes are normalized to a three-dimensional ``(z, y, x)`` view;
* tifxyz inputs are three two-dimensional ``x.tif``, ``y.tif`` and ``z.tif``
  coordinate images;
* physical voxel sizes are always specified in ``(z, y, x)`` order;
* every surface tile fetches one compact volume ROI, shared by all requested
  offsets and both normal orientations.

The public functions are also used by the command-line entry point and by
offline synthetic tests.  Zarr is imported lazily so geometry and pipeline
tests do not require a network/storage installation.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import fsspec
import numpy as np
import tifffile
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from native_surface_sampler import (
    NormalField,
    SurfaceGrid,
    estimate_surface_normals,
    sample_trilinear_zyx,
    validate_surface_geometry,
)


Image.MAX_IMAGE_PIXELS = None

SCHEMA_VERSION = 1
MAX_METADATA_BYTES = 4 * 1024 * 1024
NUMERIC_LAYER_PATTERN = re.compile(r"^(\d+)\.(?:tif|tiff|png)$", re.IGNORECASE)
SUPPORTED_OUTPUT_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "float32": np.dtype(np.float32),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_source(source: str) -> str:
    """Remove credentials/query/fragment before recording a source in a manifest."""

    parsed = urlsplit(str(source))
    if not parsed.scheme:
        return str(Path(source).expanduser())
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def is_remote_source(source: str) -> bool:
    scheme = urlsplit(str(source)).scheme.lower()
    return scheme not in ("", "file")


def join_source(source: str, filename: str) -> str:
    if is_remote_source(source):
        return source.rstrip("/") + "/" + filename
    return str(Path(source).expanduser() / filename)


def _read_source_bytes(
    source: str,
    *,
    storage_options: Mapping[str, Any] | None = None,
    maximum_bytes: int | None = None,
) -> bytes:
    options = dict(storage_options or {})
    if is_remote_source(source):
        with fsspec.open(source, mode="rb", **options) as handle:
            if maximum_bytes is None:
                return handle.read()
            data = handle.read(maximum_bytes + 1)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
        if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
            raise ValueError(f"input file exceeds the {maximum_bytes}-byte safety limit: {path}")
        data = path.read_bytes()
    if maximum_bytes is not None and len(data) > maximum_bytes:
        raise ValueError(f"input file exceeds the {maximum_bytes}-byte safety limit: {source}")
    return data


def _read_tiff_2d(
    source: str,
    *,
    storage_options: Mapping[str, Any] | None,
    maximum_pixels: int,
) -> tuple[NDArray[Any], str]:
    """Read one grayscale TIFF after inspecting dimensionality and pixel count."""

    if is_remote_source(source):
        payload = _read_source_bytes(source, storage_options=storage_options)
        digest = sha256_bytes(payload)
        source_object: Any = io.BytesIO(payload)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TIFF not found: {path}")
        digest = sha256_file(path)
        source_object = path

    with tifffile.TiffFile(source_object) as tif:
        if len(tif.series) != 1:
            raise ValueError(f"coordinate TIFF must contain exactly one image series: {source}")
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        if len(shape) != 2:
            raise ValueError(f"coordinate TIFF must be 2-D grayscale, got {shape}: {source}")
        if math.prod(shape) > int(maximum_pixels):
            raise ValueError(
                f"coordinate TIFF has {math.prod(shape)} pixels, over safety limit "
                f"{maximum_pixels}: {source}"
            )
        if not (
            np.issubdtype(series.dtype, np.integer)
            or np.issubdtype(series.dtype, np.floating)
        ):
            raise ValueError(f"coordinate TIFF must have numeric dtype, got {series.dtype}")
        array = series.asarray()
    return np.asarray(array), digest


@dataclass(frozen=True)
class TifxyzAsset:
    surface: SurfaceGrid
    metadata: Mapping[str, Any]
    scale_yx: tuple[float, float]
    source: str
    input_sha256: Mapping[str, str]
    mask_source: str
    stored_shape: tuple[int, int]
    output_shape: tuple[int, int]
    interpolation: str

    def manifest(self) -> Mapping[str, Any]:
        return {
            "source": sanitize_source(self.source),
            "stored_shape": list(self.stored_shape),
            "output_shape": list(self.output_shape),
            "scale_yx": list(self.scale_yx),
            "interpolation": self.interpolation,
            "mask_source": self.mask_source,
            "input_sha256": dict(self.input_sha256),
            "metadata": dict(self.metadata),
        }


def _parse_tifxyz_metadata(payload: bytes, source: str) -> Mapping[str, Any]:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"tifxyz metadata is not UTF-8: {source}") from exc
    try:
        metadata = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tifxyz metadata JSON at {source}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("tifxyz meta.json must contain a JSON object")
    return metadata


def _parse_scale_yx(metadata: Mapping[str, Any]) -> tuple[float, float]:
    raw = metadata.get("scale", [1.0, 1.0])
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("tifxyz metadata scale must be [x_scale, y_scale]")
    scale_x, scale_y = float(raw[0]), float(raw[1])
    if not np.isfinite((scale_x, scale_y)).all() or scale_x <= 0 or scale_y <= 0:
        raise ValueError("tifxyz scale values must be finite and positive")
    return (scale_y, scale_x)


def _derive_full_shape(
    stored_shape: tuple[int, int], scale_yx: tuple[float, float]
) -> tuple[int, int]:
    # Official tifxyz metadata commonly serializes a logical 0.05 step as the
    # float32 value 0.05000000074505806.  Blind truncation would therefore turn
    # 118 / 0.05 into 2359 instead of the official 2360-pixel raster.  Snap a
    # ratio that is numerically close to an integer before retaining the API's
    # floor behavior for genuinely fractional ratios.
    dimensions: list[int] = []
    for size, scale in zip(stored_shape, scale_yx):
        ratio = size / scale
        nearest = int(round(ratio))
        tolerance = 1e-6 * max(1.0, abs(ratio))
        dimensions.append(nearest if abs(ratio - nearest) <= tolerance else int(ratio))
    result = tuple(dimensions)
    if any(value <= 0 for value in result):
        raise ValueError(
            f"tifxyz scale {scale_yx} produces invalid full shape {result} "
            f"from stored shape {stored_shape}"
        )
    return result  # type: ignore[return-value]


def resample_surface_grid(
    surface: SurfaceGrid,
    *,
    target_shape: tuple[int, int],
    source_step_yx: tuple[float, float] | None = None,
    interpolation: str = "linear",
    row_chunk: int = 512,
) -> SurfaceGrid:
    """Mask-aware resampling from a stored tifxyz grid to an output raster.

    ``source_step_yx`` specifies how far a target output pixel advances in the
    stored coordinate images.  If omitted, endpoint-aligned steps are derived.
    Coordinates never interpolate through holes: values are normalized by an
    interpolated validity weight, and output validity requires essentially full
    local support.
    """

    target_height, target_width = (int(value) for value in target_shape)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_shape must contain positive values")
    if interpolation not in {"linear", "cubic"}:
        raise ValueError("surface interpolation must be 'linear' or 'cubic'")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    if target_shape == surface.shape:
        return surface
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError("surface resampling requires scipy") from exc

    source_height, source_width = surface.shape
    if source_step_yx is None:
        step_y = (source_height - 1) / max(target_height - 1, 1)
        step_x = (source_width - 1) / max(target_width - 1, 1)
    else:
        step_y, step_x = (float(value) for value in source_step_yx)
        if not np.isfinite((step_y, step_x)).all() or step_y <= 0 or step_x <= 0:
            raise ValueError("source_step_yx values must be finite and positive")
    order = 1 if interpolation == "linear" else 3
    source_valid = surface.valid.astype(np.float64)
    output_coordinates = [
        np.zeros(target_shape, dtype=np.float64),
        np.zeros(target_shape, dtype=np.float64),
        np.zeros(target_shape, dtype=np.float64),
    ]
    output_valid = np.zeros(target_shape, dtype=bool)
    source_arrays = (surface.x, surface.y, surface.z)
    weighted_sources = [np.where(surface.valid, array, 0.0) for array in source_arrays]

    columns = np.arange(target_width, dtype=np.float64) * step_x
    columns = np.clip(columns, 0.0, source_width - 1)
    for row_start in range(0, target_height, row_chunk):
        row_stop = min(row_start + row_chunk, target_height)
        rows = np.arange(row_start, row_stop, dtype=np.float64) * step_y
        rows = np.clip(rows, 0.0, source_height - 1)
        row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
        sampling_grid = np.stack((row_grid, column_grid), axis=0)
        weight = map_coordinates(
            source_valid,
            sampling_grid,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        nearest_valid = map_coordinates(
            source_valid,
            sampling_grid,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ) > 0.5
        chunk_valid = nearest_valid & (weight >= 1.0 - 1e-6)
        output_valid[row_start:row_stop] = chunk_valid
        for component, weighted_source in enumerate(weighted_sources):
            sampled = map_coordinates(
                weighted_source,
                sampling_grid,
                order=order,
                mode="nearest",
                prefilter=(order > 1),
            )
            sampled = np.divide(
                sampled,
                weight,
                out=np.full(sampled.shape, np.nan, dtype=np.float64),
                where=weight > 1e-12,
            )
            output_coordinates[component][row_start:row_stop] = sampled
    return SurfaceGrid.from_tifxyz(*output_coordinates, mask=output_valid)


def load_tifxyz_asset(
    source: str | Path,
    *,
    storage_options: Mapping[str, Any] | None = None,
    maximum_surface_pixels: int = 250_000_000,
    resolution: str = "stored",
    output_shape: tuple[int, int] | None = None,
    interpolation: str = "linear",
) -> TifxyzAsset:
    """Safely load local or remote tifxyz coordinate images and metadata."""

    source_string = str(source)
    if resolution not in {"stored", "full"}:
        raise ValueError("resolution must be 'stored' or 'full'")
    metadata_source = join_source(source_string, "meta.json")
    metadata_payload = _read_source_bytes(
        metadata_source,
        storage_options=storage_options,
        maximum_bytes=MAX_METADATA_BYTES,
    )
    metadata = _parse_tifxyz_metadata(metadata_payload, metadata_source)
    scale_yx = _parse_scale_yx(metadata)
    hashes: dict[str, str] = {"meta.json": sha256_bytes(metadata_payload)}

    coordinate_arrays: list[NDArray[Any]] = []
    for component in ("x", "y", "z"):
        filename = f"{component}.tif"
        array, digest = _read_tiff_2d(
            join_source(source_string, filename),
            storage_options=storage_options,
            maximum_pixels=maximum_surface_pixels,
        )
        coordinate_arrays.append(array)
        hashes[filename] = digest
    x, y, z = coordinate_arrays
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(
            f"tifxyz coordinate shapes differ: x={x.shape}, y={y.shape}, z={z.shape}"
        )
    stored_shape = tuple(int(value) for value in x.shape)

    mask_source = "derived:z>0-and-finite"
    mask_uri = join_source(source_string, "mask.tif")
    try:
        mask_array, mask_digest = _read_tiff_2d(
            mask_uri,
            storage_options=storage_options,
            maximum_pixels=maximum_surface_pixels,
        )
    except FileNotFoundError:
        mask = (np.asarray(z) > 0) & np.isfinite(z)
    else:
        if mask_array.shape != x.shape:
            raise ValueError(
                f"tifxyz mask shape {mask_array.shape} does not match coordinates {x.shape}"
            )
        mask = mask_array != 0
        hashes["mask.tif"] = mask_digest
        mask_source = "mask.tif:nonzero"

    surface = SurfaceGrid.from_tifxyz(x, y, z, mask=mask)
    if output_shape is None:
        target_shape = (
            _derive_full_shape(stored_shape, scale_yx)
            if resolution == "full"
            else stored_shape
        )
    else:
        target_shape = tuple(int(value) for value in output_shape)
        if len(target_shape) != 2 or any(value <= 0 for value in target_shape):
            raise ValueError("output_shape must contain two positive integers")
    if math.prod(target_shape) > maximum_surface_pixels:
        raise ValueError(
            f"requested output surface has {math.prod(target_shape)} pixels, over "
            f"safety limit {maximum_surface_pixels}"
        )
    if target_shape != stored_shape:
        source_step = scale_yx if resolution == "full" and output_shape is None else None
        surface = resample_surface_grid(
            surface,
            target_shape=target_shape,
            source_step_yx=source_step,
            interpolation=interpolation,
        )
    return TifxyzAsset(
        surface=surface,
        metadata=metadata,
        scale_yx=scale_yx,
        source=source_string,
        input_sha256=hashes,
        mask_source=mask_source,
        stored_shape=stored_shape,
        output_shape=surface.shape,
        interpolation=("none" if target_shape == stored_shape else interpolation),
    )


class AxisNormalizedVolume:
    """Read-only three-dimensional zyx view over an N-dimensional array."""

    def __init__(
        self,
        array: Any,
        *,
        axes: str,
        fixed_indices: Mapping[str, int] | None = None,
    ) -> None:
        normalized_axes = "".join(
            character.lower() for character in axes if character.isalpha()
        )
        if len(normalized_axes) != getattr(array, "ndim", -1):
            raise ValueError(
                f"volume axes {normalized_axes!r} do not match array ndim={getattr(array, 'ndim', None)}"
            )
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError(f"volume axes must be unique, got {normalized_axes!r}")
        if not set("zyx").issubset(normalized_axes):
            raise ValueError("volume axes must include z, y and x")
        fixed = {
            str(key).lower(): int(value)
            for key, value in (fixed_indices or {}).items()
        }
        unsupported = set(normalized_axes) - set("zyx") - set(fixed)
        if unsupported:
            raise ValueError(
                f"fixed indices are required for non-spatial axes: {sorted(unsupported)}"
            )
        for axis, index in fixed.items():
            if axis not in normalized_axes:
                raise ValueError(f"fixed index supplied for absent volume axis {axis!r}")
            size = int(array.shape[normalized_axes.index(axis)])
            if index < 0 or index >= size:
                raise IndexError(f"fixed {axis} index {index} is outside [0, {size})")

        self.array = array
        self.axes = normalized_axes
        self.fixed_indices = fixed
        self.shape = tuple(int(array.shape[normalized_axes.index(axis)]) for axis in "zyx")
        self.ndim = 3
        self.dtype = np.dtype(array.dtype)
        chunks = getattr(array, "chunks", None)
        self.chunks = (
            tuple(int(chunks[normalized_axes.index(axis)]) for axis in "zyx")
            if chunks is not None
            else None
        )
        self.read_count = 0

    def __getitem__(self, key: Any) -> NDArray[Any]:
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError("AxisNormalizedVolume requires exactly three z,y,x indices")
        spatial_selection = dict(zip("zyx", key))
        source_selection: list[Any] = []
        remaining_axes: list[str] = []
        for axis in self.axes:
            if axis in spatial_selection:
                source_selection.append(spatial_selection[axis])
                remaining_axes.append(axis)
            else:
                source_selection.append(self.fixed_indices[axis])
        self.read_count += 1
        data = np.asarray(self.array[tuple(source_selection)])
        if data.ndim != 3:
            raise ValueError(f"normalized volume read returned shape {data.shape}, expected 3-D")
        permutation = tuple(remaining_axes.index(axis) for axis in "zyx")
        if permutation != (0, 1, 2):
            data = np.transpose(data, permutation)
        return data


@dataclass(frozen=True)
class OpenedVolume:
    volume_zyx: AxisNormalizedVolume
    source: str
    array_path: str
    axes: str
    metadata: Mapping[str, Any]

    def manifest(self) -> Mapping[str, Any]:
        return {
            "source": sanitize_source(self.source),
            "array_path": self.array_path,
            "source_axes": self.axes,
            "normalized_axes": "zyx",
            "shape_zyx": list(self.volume_zyx.shape),
            "chunks_zyx": (
                list(self.volume_zyx.chunks) if self.volume_zyx.chunks is not None else None
            ),
            "dtype": str(self.volume_zyx.dtype),
            "metadata": dict(self.metadata),
        }


def _attrs_dict(node: Any) -> Mapping[str, Any]:
    attrs = getattr(node, "attrs", None)
    if attrs is None:
        return {}
    try:
        return dict(attrs)
    except Exception:
        return {}


def _resolve_zarr_array(root: Any, array_path: str | None) -> tuple[Any, str, Mapping[str, Any]]:
    root_attrs = _attrs_dict(root)
    if array_path:
        return root[array_path], array_path, root_attrs
    if hasattr(root, "shape") and hasattr(root, "dtype"):
        return root, "", root_attrs
    multiscales = root_attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets and isinstance(datasets[0], dict) and "path" in datasets[0]:
            path = str(datasets[0]["path"])
            return root[path], path, root_attrs
    try:
        if "0" in root:
            return root["0"], "0", root_attrs
    except TypeError:
        pass
    raise ValueError(
        "could not select a Zarr array; pass --volume-array-path (usually 0 for OME-Zarr)"
    )


def _infer_axes(
    array: Any,
    root_metadata: Mapping[str, Any],
    axes_override: str | None,
) -> str:
    if axes_override and axes_override.lower() != "auto":
        return axes_override.lower()
    array_dimensions = _attrs_dict(array).get("_ARRAY_DIMENSIONS")
    if isinstance(array_dimensions, (list, tuple)):
        return "".join(str(value).lower() for value in array_dimensions)
    multiscales = root_metadata.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        axes = multiscales[0].get("axes")
        if isinstance(axes, list):
            names = [item.get("name") if isinstance(item, dict) else item for item in axes]
            if all(name is not None for name in names):
                return "".join(str(name).lower() for name in names)
    raise ValueError(
        "OME-Zarr axes metadata is absent; explicitly pass --volume-axes zyx"
    )


def open_volume_zyx(
    source: str | Path,
    *,
    array_path: str | None = None,
    axes: str | None = "auto",
    fixed_indices: Mapping[str, int] | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> OpenedVolume:
    """Open local/remote OME-Zarr (or local NPY for offline testing) as zyx."""

    source_string = str(source)
    source_path = Path(source_string).expanduser() if not is_remote_source(source_string) else None
    if source_path is not None and source_path.suffix.lower() == ".npy":
        array = np.load(source_path, mmap_mode="r")
        chosen_path = ""
        root_metadata: Mapping[str, Any] = {}
    else:
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "opening OME-Zarr requires the optional 'zarr' package; "
                "install zarr and the appropriate fsspec backend (for example s3fs)"
            ) from exc
        options = dict(storage_options or {})
        if is_remote_source(source_string):
            try:
                root = zarr.open(source_string, mode="r", storage_options=options)
            except Exception as direct_error:
                try:
                    mapper = fsspec.get_mapper(source_string, **options)
                    root = zarr.open(mapper, mode="r")
                except Exception as mapper_error:
                    raise RuntimeError(
                        f"failed to open remote Zarr {sanitize_source(source_string)}: "
                        f"direct={direct_error}; mapper={mapper_error}"
                    ) from mapper_error
        else:
            root = zarr.open(str(source_path), mode="r")
        array, chosen_path, root_metadata = _resolve_zarr_array(root, array_path)

    inferred_axes = _infer_axes(array, root_metadata, axes)
    normalized = AxisNormalizedVolume(
        array, axes=inferred_axes, fixed_indices=fixed_indices
    )
    return OpenedVolume(
        volume_zyx=normalized,
        source=source_string,
        array_path=chosen_path,
        axes=inferred_axes,
        metadata=root_metadata,
    )


@dataclass(frozen=True)
class RenderOptions:
    """Immutable native-volume rendering parameters."""

    voxel_size_zyx_um: tuple[float, float, float]
    offsets_um: tuple[float, ...]
    signs: tuple[str, ...] = ("positive", "negative")
    tile_shape: tuple[int, int] = (512, 512)
    output_dtype: str = "uint8"
    fill_value: float = 0.0
    png_mode: str = "preview"
    overwrite: bool = False
    hash_outputs: bool = False

    def __post_init__(self) -> None:
        voxel_sizes = np.asarray(self.voxel_size_zyx_um, dtype=np.float64)
        if voxel_sizes.shape != (3,) or not np.isfinite(voxel_sizes).all() or np.any(
            voxel_sizes <= 0
        ):
            raise ValueError("voxel_size_zyx_um must be three finite positive values")
        offsets = np.asarray(self.offsets_um, dtype=np.float64)
        if offsets.ndim != 1 or offsets.size == 0 or not np.isfinite(offsets).all():
            raise ValueError("offsets_um must be a non-empty finite sequence")
        if len(set(self.signs)) != len(self.signs) or not self.signs:
            raise ValueError("one or more unique normal signs are required")
        if any(sign not in {"positive", "negative"} for sign in self.signs):
            raise ValueError("signs may contain only 'positive' and 'negative'")
        if len(self.tile_shape) != 2 or any(int(value) <= 0 for value in self.tile_shape):
            raise ValueError("tile_shape must contain two positive integers")
        if self.output_dtype not in SUPPORTED_OUTPUT_DTYPES:
            raise ValueError(
                f"output_dtype must be one of {sorted(SUPPORTED_OUTPUT_DTYPES)}, "
                f"got {self.output_dtype!r}"
            )
        if self.png_mode not in {"none", "preview", "all"}:
            raise ValueError("png_mode must be 'none', 'preview' or 'all'")

    @property
    def numpy_output_dtype(self) -> np.dtype[Any]:
        return SUPPORTED_OUTPUT_DTYPES[self.output_dtype]

    def manifest(self) -> Mapping[str, Any]:
        return {
            "voxel_size_zyx_um": list(self.voxel_size_zyx_um),
            "offsets_um": list(self.offsets_um),
            "frame_count": len(self.offsets_um),
            "signs": list(self.signs),
            "tile_shape": list(self.tile_shape),
            "output_dtype": self.output_dtype,
            "fill_value": self.fill_value,
            "png_mode": self.png_mode,
        }


@dataclass
class RoiFetchStats:
    tiles_total: int = 0
    tiles_fetched: int = 0
    tiles_without_samples: int = 0
    roi_voxels_fetched: int = 0
    roi_bytes_fetched: int = 0
    max_roi_shape_zyx: tuple[int, int, int] = (0, 0, 0)
    elapsed_seconds: float = 0.0

    def record(self, roi: NDArray[Any]) -> None:
        self.tiles_fetched += 1
        self.roi_voxels_fetched += int(roi.size)
        self.roi_bytes_fetched += int(roi.nbytes)
        shape = tuple(int(value) for value in roi.shape)
        if math.prod(shape) > math.prod(self.max_roi_shape_zyx):
            self.max_roi_shape_zyx = shape  # type: ignore[assignment]

    def manifest(self) -> Mapping[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    """Convert NaN/inf values in metric reports to JSON null."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _prepare_output_directory(output_directory: Path, options: RenderOptions) -> None:
    if output_directory.exists():
        existing = list(output_directory.iterdir())
        if existing and not options.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_directory}; pass --overwrite "
                "to replace renderer-managed outputs"
            )
        if options.overwrite:
            # Delete only paths owned by this renderer; unrelated user files are
            # preserved and will make a later manual audit obvious.
            for sign in ("positive", "negative"):
                sign_directory = output_directory / sign
                if sign_directory.exists():
                    shutil.rmtree(sign_directory)
            for filename in ("manifest.json", "geometry-qc.json", "calibration.json"):
                managed = output_directory / filename
                if managed.exists():
                    managed.unlink()
    output_directory.mkdir(parents=True, exist_ok=True)


def _cast_output_frame(
    frame: NDArray[np.floating[Any]],
    *,
    valid: NDArray[np.bool_],
    dtype: np.dtype[Any],
    fill_value: float,
) -> NDArray[Any]:
    result = np.full(frame.shape, fill_value, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        converted = np.rint(np.clip(frame, limits.min, limits.max)).astype(dtype)
    else:
        converted = frame.astype(dtype, copy=False)
    result[valid] = converted[valid]
    return result


def _png_u8(array: NDArray[Any]) -> tuple[NDArray[np.uint8], str]:
    if array.dtype == np.uint8:
        return array, "identity-uint8"
    if array.dtype == np.uint16:
        return np.right_shift(array, 8).astype(np.uint8), "uint16-right-shift-8"
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8), "all-nonfinite-to-zero"
    values = np.asarray(array[finite], dtype=np.float64)
    low, high = np.percentile(values, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((np.asarray(array, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8), f"percentile-1-99.5:{low:g}:{high:g}"


class _SignLayerWriter:
    def __init__(
        self,
        directory: Path,
        *,
        shape: tuple[int, int],
        frame_count: int,
        dtype: np.dtype[Any],
        fill_value: float,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.shape = shape
        self.frame_count = frame_count
        self.dtype = dtype
        self.layer_width = max(2, len(str(frame_count - 1)))
        self.layer_paths = [
            directory / f"{index:0{self.layer_width}d}.tif"
            for index in range(frame_count)
        ]
        self.layers: list[NDArray[Any]] = []
        for path in self.layer_paths:
            layer = tifffile.memmap(
                path,
                shape=shape,
                dtype=dtype,
                photometric="minisblack",
                bigtiff=(math.prod(shape) * dtype.itemsize >= 2**32),
            )
            layer[...] = fill_value
            self.layers.append(layer)
        count_dtype = np.uint16 if frame_count > 255 else np.uint8
        self.valid_count = np.zeros(shape, dtype=count_dtype)

    def write_tile(
        self,
        frame_index: int,
        row_slice: slice,
        column_slice: slice,
        values: NDArray[np.floating[Any]],
        valid: NDArray[np.bool_],
        *,
        fill_value: float,
    ) -> None:
        self.layers[frame_index][row_slice, column_slice] = _cast_output_frame(
            values, valid=valid, dtype=self.dtype, fill_value=fill_value
        )
        self.valid_count[row_slice, column_slice] += valid.astype(self.valid_count.dtype)

    def finalize(
        self,
        *,
        png_mode: str,
        hash_outputs: bool,
    ) -> Mapping[str, Any]:
        for layer in self.layers:
            layer.flush()
        del self.layers[:]
        valid_any = self.valid_count > 0
        valid_all = self.valid_count == self.frame_count
        valid_any_path = self.directory / "valid-any.tif"
        valid_all_path = self.directory / "valid-all.tif"
        tifffile.imwrite(valid_any_path, valid_any.astype(np.uint8) * 255)
        tifffile.imwrite(valid_all_path, valid_all.astype(np.uint8) * 255)

        if png_mode == "none":
            png_indices: list[int] = []
        elif png_mode == "all":
            png_indices = list(range(self.frame_count))
        else:
            png_indices = sorted({0, (self.frame_count - 1) // 2, self.frame_count - 1})
        png_directory = self.directory / "png"
        png_records: list[dict[str, Any]] = []
        if png_indices:
            png_directory.mkdir(exist_ok=True)
        for index in png_indices:
            array = tifffile.imread(self.layer_paths[index])
            png_array, conversion = _png_u8(array)
            path = png_directory / f"{index:0{self.layer_width}d}.png"
            Image.fromarray(png_array, mode="L").save(path, optimize=True)
            png_records.append(
                {
                    "frame": index,
                    "path": str(path.relative_to(self.directory.parent)),
                    "conversion": conversion,
                    "bytes": path.stat().st_size,
                }
            )

        layer_records: list[dict[str, Any]] = []
        for index, path in enumerate(self.layer_paths):
            record: dict[str, Any] = {
                "frame": index,
                "path": str(path.relative_to(self.directory.parent)),
                "bytes": path.stat().st_size,
            }
            if hash_outputs:
                record["sha256"] = sha256_file(path)
            layer_records.append(record)
        return {
            "layers": layer_records,
            "png": png_records,
            "valid_any_path": str(valid_any_path.relative_to(self.directory.parent)),
            "valid_all_path": str(valid_all_path.relative_to(self.directory.parent)),
            "valid_any_fraction": float(valid_any.mean()),
            "valid_all_fraction": float(valid_all.mean()),
            "valid_sample_fraction": float(
                self.valid_count.sum(dtype=np.float64)
                / (self.frame_count * math.prod(self.shape))
            ),
        }


def iter_surface_tiles(
    shape: tuple[int, int], tile_shape: tuple[int, int]
) -> Iterator[tuple[slice, slice]]:
    height, width = shape
    tile_height, tile_width = tile_shape
    for row_start in range(0, height, tile_height):
        row_slice = slice(row_start, min(row_start + tile_height, height), 1)
        for column_start in range(0, width, tile_width):
            column_slice = slice(
                column_start, min(column_start + tile_width, width), 1
            )
            yield row_slice, column_slice


def _global_bounds_mask(
    points_xyz: NDArray[np.float64], volume_shape_zyx: tuple[int, int, int]
) -> NDArray[np.bool_]:
    z_size, y_size, x_size = volume_shape_zyx
    return (
        np.isfinite(points_xyz).all(axis=-1)
        & (points_xyz[..., 0] >= 0.0)
        & (points_xyz[..., 0] <= x_size - 1)
        & (points_xyz[..., 1] >= 0.0)
        & (points_xyz[..., 1] <= y_size - 1)
        & (points_xyz[..., 2] >= 0.0)
        & (points_xyz[..., 2] <= z_size - 1)
    )


def _tile_roi_bounds_zyx(
    base_xyz: NDArray[np.float64],
    positive_normal_xyz: NDArray[np.float64],
    base_valid: NDArray[np.bool_],
    *,
    volume_shape_zyx: tuple[int, int, int],
    voxel_size_xyz_um: NDArray[np.float64],
    offsets_um: Sequence[float],
    signs: Sequence[str],
) -> tuple[slice, slice, slice] | None:
    minimum_xyz = np.full(3, np.inf, dtype=np.float64)
    maximum_xyz = np.full(3, -np.inf, dtype=np.float64)
    found = False
    for sign_name in signs:
        sign = 1.0 if sign_name == "positive" else -1.0
        normal_voxels_per_um = sign * positive_normal_xyz / voxel_size_xyz_um
        for offset in offsets_um:
            points = base_xyz + float(offset) * normal_voxels_per_um
            usable = base_valid & _global_bounds_mask(points, volume_shape_zyx)
            if np.any(usable):
                selected = points[usable]
                minimum_xyz = np.minimum(minimum_xyz, np.min(selected, axis=0))
                maximum_xyz = np.maximum(maximum_xyz, np.max(selected, axis=0))
                found = True
    if not found:
        return None
    z_size, y_size, x_size = volume_shape_zyx
    lower_xyz = np.floor(minimum_xyz).astype(np.int64)
    upper_xyz = np.ceil(maximum_xyz).astype(np.int64)
    lower_xyz = np.maximum(lower_xyz, 0)
    upper_xyz = np.minimum(upper_xyz, np.asarray((x_size - 1, y_size - 1, z_size - 1)))
    x_slice = slice(int(lower_xyz[0]), int(upper_xyz[0]) + 1, 1)
    y_slice = slice(int(lower_xyz[1]), int(upper_xyz[1]) + 1, 1)
    z_slice = slice(int(lower_xyz[2]), int(upper_xyz[2]) + 1, 1)
    return z_slice, y_slice, x_slice


def render_surface_to_directory(
    volume_zyx: Any,
    surface: SurfaceGrid,
    output_directory: str | Path,
    *,
    options: RenderOptions,
    tifxyz_manifest: Mapping[str, Any] | None = None,
    volume_manifest: Mapping[str, Any] | None = None,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Render every tile while fetching one shared compact 3-D ROI per tile."""

    if getattr(volume_zyx, "ndim", None) != 3:
        raise ValueError("volume_zyx must be explicitly normalized to three z,y,x axes")
    volume_shape = tuple(int(value) for value in volume_zyx.shape)
    if len(volume_shape) != 3 or any(value <= 0 for value in volume_shape):
        raise ValueError(f"invalid z,y,x volume shape: {volume_shape}")
    output_path = Path(output_directory)
    _prepare_output_directory(output_path, options)
    geometry_reports: dict[str, Any] = {}
    for sign_name in options.signs:
        report = validate_surface_geometry(
            surface,
            volume_shape_zyx=volume_shape,
            voxel_size_zyx_um=options.voxel_size_zyx_um,
            normal_offsets_um=options.offsets_um,
            normal_sign=(1 if sign_name == "positive" else -1),
        )
        geometry_reports[sign_name] = _json_safe(report.to_dict())
    _atomic_json(
        output_path / "geometry-qc.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "reports_by_normal_sign": geometry_reports,
        },
    )

    running_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_utc": utc_now(),
        "coordinate_conventions": {
            "volume_index_order": "z,y,x",
            "tifxyz_input_order": "separate x,y,z coordinate images",
            "point_component_order": "x,y,z",
            "voxel_size_order": "z,y,x",
            "output_stack_order": "frame,row,column",
        },
        "tifxyz": dict(tifxyz_manifest or {}),
        "volume": dict(volume_manifest or {}),
        "render_options": dict(options.manifest()),
        "geometry_qc_path": "geometry-qc.json",
    }
    _atomic_json(output_path / "manifest.json", _json_safe(running_manifest))

    normal_field: NormalField = estimate_surface_normals(
        surface, voxel_size_zyx_um=options.voxel_size_zyx_um
    )
    # Rendering needs only one orientation; the second is generated with a
    # scalar sign.  Drop the explicitly materialized negative field before
    # entering the tile loop to save 3 * H * W * 8 bytes on large surfaces.
    positive_normals = normal_field.positive_xyz
    normals_valid = normal_field.valid
    del normal_field
    voxel_size_xyz_um = np.asarray(options.voxel_size_zyx_um, dtype=np.float64)[::-1]
    writers = {
        sign: _SignLayerWriter(
            output_path / sign,
            shape=surface.shape,
            frame_count=len(options.offsets_um),
            dtype=options.numpy_output_dtype,
            fill_value=options.fill_value,
        )
        for sign in options.signs
    }

    tiles = list(iter_surface_tiles(surface.shape, options.tile_shape))
    stats = RoiFetchStats(tiles_total=len(tiles))
    started = time.monotonic()
    for tile_index, (row_slice, column_slice) in enumerate(tiles, start=1):
        tile_key = (row_slice, column_slice)
        # Stack only this tile.  Calling SurfaceGrid.points_xyz before slicing
        # would allocate the complete H*W*3 surface once per tile.
        base_xyz = np.stack(
            (
                surface.x[tile_key],
                surface.y[tile_key],
                surface.z[tile_key],
            ),
            axis=-1,
        )
        positive_normal = positive_normals[tile_key]
        base_valid = surface.valid[tile_key] & normals_valid[tile_key]
        roi_slices = _tile_roi_bounds_zyx(
            base_xyz,
            positive_normal,
            base_valid,
            volume_shape_zyx=volume_shape,
            voxel_size_xyz_um=voxel_size_xyz_um,
            offsets_um=options.offsets_um,
            signs=options.signs,
        )
        if roi_slices is None:
            stats.tiles_without_samples += 1
            if progress is not None:
                progress(tile_index, len(tiles), {"fetched": False})
            continue
        z_slice, y_slice, x_slice = roi_slices
        # This is the only source-volume read for the tile.  Both normal signs
        # and all frames below sample this same in-memory compact ROI.
        roi = np.asarray(volume_zyx[z_slice, y_slice, x_slice])
        expected_shape = (
            z_slice.stop - z_slice.start,
            y_slice.stop - y_slice.start,
            x_slice.stop - x_slice.start,
        )
        if roi.shape != expected_shape:
            raise ValueError(
                f"volume ROI read returned shape {roi.shape}, expected {expected_shape}"
            )
        stats.record(roi)
        roi_origin_xyz = np.asarray(
            (x_slice.start, y_slice.start, z_slice.start), dtype=np.float64
        )
        for sign_name, writer in writers.items():
            sign = 1.0 if sign_name == "positive" else -1.0
            normal_voxels_per_um = sign * positive_normal / voxel_size_xyz_um
            for frame_index, offset_um in enumerate(options.offsets_um):
                global_points = base_xyz + float(offset_um) * normal_voxels_per_um
                global_valid = base_valid & _global_bounds_mask(global_points, volume_shape)
                local_points = global_points - roi_origin_xyz
                sample = sample_trilinear_zyx(
                    roi,
                    local_points,
                    point_valid=global_valid,
                    fill_value=options.fill_value,
                    output_dtype=np.float32,
                )
                writer.write_tile(
                    frame_index,
                    row_slice,
                    column_slice,
                    sample.values,
                    sample.valid,
                    fill_value=options.fill_value,
                )
        if progress is not None:
            progress(
                tile_index,
                len(tiles),
                {
                    "fetched": True,
                    "roi_shape_zyx": list(roi.shape),
                    "roi_bytes": int(roi.nbytes),
                },
            )

    stats.elapsed_seconds = float(time.monotonic() - started)
    output_records = {
        sign: writer.finalize(
            png_mode=options.png_mode, hash_outputs=options.hash_outputs
        )
        for sign, writer in writers.items()
    }
    complete_manifest = {
        **running_manifest,
        "status": "complete",
        "completed_utc": utc_now(),
        "render_statistics": stats.manifest(),
        "outputs_by_normal_sign": output_records,
    }
    _atomic_json(output_path / "manifest.json", _json_safe(complete_manifest))
    return _json_safe(complete_manifest)


@dataclass(frozen=True)
class CalibrationThresholds:
    valid_mask_iou: float = 0.995
    median_layer_correlation: float = 0.995
    median_absolute_difference: float = 2.0
    erosion_pixels: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.valid_mask_iou <= 1.0:
            raise ValueError("valid_mask_iou threshold must be in [0, 1]")
        if not -1.0 <= self.median_layer_correlation <= 1.0:
            raise ValueError("correlation threshold must be in [-1, 1]")
        if self.median_absolute_difference < 0.0:
            raise ValueError("median_absolute_difference threshold must be non-negative")
        if int(self.erosion_pixels) != self.erosion_pixels or self.erosion_pixels < 0:
            raise ValueError("erosion_pixels must be a non-negative integer")


def list_numeric_layers(directory: str | Path) -> list[Path]:
    """Return a gap-free, numerically sorted layer sequence."""

    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"surface stack directory not found: {path}")
    by_index: dict[int, Path] = {}
    for candidate in path.iterdir():
        match = NUMERIC_LAYER_PATTERN.match(candidate.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in by_index:
            raise ValueError(
                f"duplicate numeric layer index {index}: {by_index[index]} and {candidate}"
            )
        by_index[index] = candidate
    if not by_index:
        raise FileNotFoundError(f"no numeric TIFF/PNG layers found in {path}")
    indices = sorted(by_index)
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(f"numeric layer sequence has gaps: found {indices}")
    return [by_index[index] for index in indices]


def _read_layer_2d(path: Path) -> NDArray[Any]:
    if path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"calibration layer must be 2-D grayscale, got {array.shape}: {path}")
    return array


def _binary_erode(mask: NDArray[np.bool_], iterations: int) -> NDArray[np.bool_]:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        next_result = np.ones(result.shape, dtype=bool)
        for row_offset in range(3):
            for column_offset in range(3):
                next_result &= padded[
                    row_offset : row_offset + result.shape[0],
                    column_offset : column_offset + result.shape[1],
                ]
        result = next_result
    return result


def _pearson_correlation(a: NDArray[Any], b: NDArray[Any]) -> float:
    a_values = np.asarray(a, dtype=np.float64).reshape(-1)
    b_values = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a_values) & np.isfinite(b_values)
    a_values = a_values[finite]
    b_values = b_values[finite]
    if a_values.size < 2:
        return float("nan")
    a_centered = a_values - np.mean(a_values)
    b_centered = b_values - np.mean(b_values)
    denominator = float(
        np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered))
    )
    if denominator <= 0.0:
        return 1.0 if np.array_equal(a_values, b_values) else 0.0
    return float(np.sum(a_centered * b_centered) / denominator)


def _stack_nonzero_union(paths: Sequence[Path]) -> tuple[NDArray[np.bool_], tuple[int, int]]:
    union: NDArray[np.bool_] | None = None
    shape: tuple[int, int] | None = None
    for path in paths:
        array = _read_layer_2d(path)
        if shape is None:
            shape = tuple(int(value) for value in array.shape)
            union = np.zeros(shape, dtype=bool)
        elif array.shape != shape:
            raise ValueError(
                f"calibration stack layer shapes differ: expected {shape}, "
                f"got {array.shape} at {path}"
            )
        assert union is not None
        union |= np.isfinite(array) & (array != 0)
    assert union is not None and shape is not None
    return union, shape


def _orientation_metrics(
    published_paths: Sequence[Path],
    generated_paths: Sequence[Path],
    *,
    generated_valid: NDArray[np.bool_],
    published_valid: NDArray[np.bool_],
    reversed_frames: bool,
    erosion_pixels: int,
) -> Mapping[str, Any]:
    intersection = generated_valid & published_valid
    union = generated_valid | published_valid
    mask_iou = float(intersection.sum() / union.sum()) if np.any(union) else 1.0
    interior = _binary_erode(intersection, erosion_pixels)
    correlations: list[float] = []
    layer_mads: list[float] = []
    layer_metrics: list[dict[str, Any]] = []
    for published_index, published_path in enumerate(published_paths):
        generated_index = (
            len(generated_paths) - 1 - published_index
            if reversed_frames
            else published_index
        )
        generated_path = generated_paths[generated_index]
        published = _read_layer_2d(published_path)
        generated = _read_layer_2d(generated_path)
        if published.shape != generated.shape:
            raise ValueError(
                f"calibration shape mismatch: published {published.shape}, "
                f"generated {generated.shape}"
            )
        usable = interior & np.isfinite(published) & np.isfinite(generated)
        if np.any(usable):
            correlation = _pearson_correlation(published[usable], generated[usable])
            mad = float(
                np.median(
                    np.abs(
                        np.asarray(published[usable], dtype=np.float64)
                        - np.asarray(generated[usable], dtype=np.float64)
                    )
                )
            )
        else:
            correlation = float("nan")
            mad = float("nan")
        correlations.append(correlation)
        layer_mads.append(mad)
        layer_metrics.append(
            {
                "published_frame": published_index,
                "generated_frame": generated_index,
                "interior_pixels": int(np.count_nonzero(usable)),
                "correlation": correlation,
                "median_absolute_difference": mad,
            }
        )
    finite_correlations = np.asarray(correlations)[np.isfinite(correlations)]
    finite_mads = np.asarray(layer_mads)[np.isfinite(layer_mads)]
    return {
        "valid_mask_iou": mask_iou,
        "intersection_pixels": int(np.count_nonzero(intersection)),
        "interior_pixels": int(np.count_nonzero(interior)),
        "median_layer_correlation": (
            float(np.median(finite_correlations))
            if finite_correlations.size
            else float("nan")
        ),
        "minimum_layer_correlation": (
            float(np.min(finite_correlations))
            if finite_correlations.size
            else float("nan")
        ),
        "median_absolute_difference": (
            float(np.median(finite_mads)) if finite_mads.size else float("nan")
        ),
        "maximum_layer_mad": (
            float(np.max(finite_mads)) if finite_mads.size else float("nan")
        ),
        "layers": layer_metrics,
    }


def compare_published_stack(
    output_directory: str | Path,
    published_stack_directory: str | Path,
    *,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
    expected_orientation: str = "any",
    write_result: bool = True,
) -> Mapping[str, Any]:
    """Compare rendered signs/orderings with a published surface stack.

    Four orientations are evaluated when both signs exist: positive/negative,
    each with direct/reversed frame order.  With symmetric offsets, positive
    direct and negative reversed are mathematically identical; such ties are
    reported rather than pretending the absolute direction was resolved.
    """

    valid_expected = {
        "any",
        "positive-direct",
        "positive-reversed",
        "negative-direct",
        "negative-reversed",
    }
    if expected_orientation not in valid_expected:
        raise ValueError(
            f"expected_orientation must be one of {sorted(valid_expected)}"
        )
    output_path = Path(output_directory)
    published_paths = list_numeric_layers(published_stack_directory)
    published_valid, published_shape = _stack_nonzero_union(published_paths)
    candidates: dict[str, Mapping[str, Any]] = {}
    for sign_name in ("positive", "negative"):
        sign_directory = output_path / sign_name
        if not sign_directory.is_dir():
            continue
        generated_paths = list_numeric_layers(sign_directory)
        if len(generated_paths) != len(published_paths):
            raise ValueError(
                f"calibration frame count mismatch: published={len(published_paths)}, "
                f"{sign_name}={len(generated_paths)}"
            )
        generated_valid_path = sign_directory / "valid-any.tif"
        if not generated_valid_path.is_file():
            raise FileNotFoundError(f"generated validity mask missing: {generated_valid_path}")
        generated_valid = np.asarray(tifffile.imread(generated_valid_path)) != 0
        if generated_valid.shape != published_shape:
            raise ValueError(
                f"calibration mask shape mismatch: published={published_shape}, "
                f"generated={generated_valid.shape}"
            )
        for order_name, reversed_frames in (("direct", False), ("reversed", True)):
            orientation = f"{sign_name}-{order_name}"
            candidates[orientation] = _orientation_metrics(
                published_paths,
                generated_paths,
                generated_valid=generated_valid,
                published_valid=published_valid,
                reversed_frames=reversed_frames,
                erosion_pixels=thresholds.erosion_pixels,
            )
    if not candidates:
        raise FileNotFoundError("no positive or negative generated stack found")

    def candidate_score(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, int]:
        orientation, metrics = item
        correlation = float(metrics["median_layer_correlation"])
        mad = float(metrics["median_absolute_difference"])
        iou = float(metrics["valid_mask_iou"])
        # Deterministic tie preference is explicit, not evidence of direction.
        tie_order = list(sorted(candidates)).index(orientation)
        return (
            correlation if np.isfinite(correlation) else -np.inf,
            -mad if np.isfinite(mad) else -np.inf,
            iou,
            -tie_order,
        )

    best_orientation, best_metrics = max(candidates.items(), key=candidate_score)
    best_score = candidate_score((best_orientation, best_metrics))[:3]
    tied_orientations = [
        orientation
        for orientation, metrics in candidates.items()
        if np.allclose(
            candidate_score((orientation, metrics))[:3],
            best_score,
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )
    ]
    selected_orientation = (
        best_orientation if expected_orientation == "any" else expected_orientation
    )
    if selected_orientation not in candidates:
        raise ValueError(
            f"required orientation {selected_orientation!r} was not rendered; "
            f"available={sorted(candidates)}"
        )
    selected = candidates[selected_orientation]
    numeric_pass = (
        float(selected["valid_mask_iou"]) >= thresholds.valid_mask_iou
        and float(selected["median_layer_correlation"])
        >= thresholds.median_layer_correlation
        and float(selected["median_absolute_difference"])
        <= thresholds.median_absolute_difference
    )
    orientation_pass = (
        expected_orientation == "any" or selected_orientation in tied_orientations
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "published_stack": sanitize_source(str(published_stack_directory)),
        "published_frame_count": len(published_paths),
        "published_shape": list(published_shape),
        "thresholds": asdict(thresholds),
        "expected_orientation": expected_orientation,
        "best_orientation": best_orientation,
        "selected_orientation": selected_orientation,
        "best_orientation_ties": sorted(tied_orientations),
        "orientation_resolved": len(tied_orientations) == 1,
        "numeric_thresholds_passed": bool(numeric_pass),
        "orientation_requirement_passed": bool(orientation_pass),
        "passed": bool(numeric_pass and orientation_pass),
        "candidates": candidates,
    }
    safe_result = _json_safe(result)
    if write_result:
        _atomic_json(output_path / "calibration.json", safe_result)
        manifest_path = output_path / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["calibration"] = {
                "path": "calibration.json",
                "passed": safe_result["passed"],
                "selected_orientation": selected_orientation,
                "best_orientation": best_orientation,
                "orientation_resolved": safe_result["orientation_resolved"],
            }
            _atomic_json(manifest_path, manifest)
    return safe_result


__all__ = [
    "AxisNormalizedVolume",
    "CalibrationThresholds",
    "OpenedVolume",
    "RenderOptions",
    "RoiFetchStats",
    "TifxyzAsset",
    "compare_published_stack",
    "iter_surface_tiles",
    "list_numeric_layers",
    "load_tifxyz_asset",
    "open_volume_zyx",
    "render_surface_to_directory",
    "resample_surface_grid",
    "sanitize_source",
]
