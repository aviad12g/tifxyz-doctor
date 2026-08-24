#!/usr/bin/env python3
"""Fetch a minimal raw-CT cube and render orthogonal QC aids.

The official raw volume is a Zarr-v2 group.  This tool reads and hashes the
group/array metadata, computes the exact set of source chunks intersecting a
bounded cube, downloads each source chunk once, assembles the requested ROI,
and writes lossless raw slices plus windowed contact sheets.  It performs no
ink inference and assigns no semantic interpretation to the CT signal.

The decoder follows the array metadata.  Uncompressed chunks are copied
directly; Blosc-compressed arrays, when encountered, are decoded with VC3D's
bundled and hashed libblosc rather than an unpinned Python codec.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from sample_m7_seed_chunk import DEFAULT_BLOSC


DEFAULT_RAW_ROOT = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
    "PHerc1447/volumes/"
    "20250521151220-8.640um-1.2m-116keV-masked.zarr"
)
DEFAULT_ARRAY_PATH = "0"
DEFAULT_XYZ = (3525, 4191, 14071)
DEFAULT_RADIUS = 32
DEFAULT_VOXEL_UM = 8.64
DEFAULT_OFFSETS = (-24, -16, -8, 0, 8, 16, 24)
DEFAULT_OUTPUT = Path(
    "outputs/first-letters-geometry/vc3d-interactive/"
    "priority1-seed-3526-4188-14072"
)
DEFAULT_SEED_EVIDENCE = Path(
    "work/first-letters/vc3d-projects/pherc1447-interactive/"
    "priority-seed-m7-selection.json"
)


@dataclass(frozen=True)
class ZarrV2ArraySpec:
    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]
    dtype_string: str
    order: str
    fill_value: int | float | None
    dimension_separator: str
    compressor: Mapping[str, Any] | None
    filters: tuple[Mapping[str, Any], ...] | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ZarrV2ArraySpec":
        if metadata.get("zarr_format") != 2:
            raise ValueError("only Zarr v2 arrays are supported")
        shape = tuple(int(value) for value in metadata.get("shape", ()))
        chunks = tuple(int(value) for value in metadata.get("chunks", ()))
        if len(shape) != 3 or len(chunks) != 3:
            raise ValueError("raw array must have exactly three z/y/x dimensions")
        if any(value <= 0 for value in shape + chunks):
            raise ValueError("shape and chunks must be positive")
        order = str(metadata.get("order", ""))
        if order not in {"C", "F"}:
            raise ValueError("Zarr order must be C or F")
        separator = str(metadata.get("dimension_separator", "."))
        if separator not in {"/", "."}:
            raise ValueError("dimension_separator must be '/' or '.'")
        dtype_string = str(metadata.get("dtype", ""))
        dtype = np.dtype(dtype_string)
        if dtype.hasobject:
            raise ValueError("object arrays are unsupported")
        raw_filters = metadata.get("filters")
        filters = (
            tuple(dict(value) for value in raw_filters)
            if raw_filters is not None
            else None
        )
        if filters:
            raise ValueError("filtered Zarr chunks are unsupported")
        raw_compressor = metadata.get("compressor")
        compressor = dict(raw_compressor) if raw_compressor is not None else None
        if compressor is not None and compressor.get("id") != "blosc":
            raise ValueError(
                f"unsupported Zarr compressor: {compressor.get('id')!r}"
            )
        return cls(
            shape_zyx=shape,  # type: ignore[arg-type]
            chunks_zyx=chunks,  # type: ignore[arg-type]
            dtype_string=dtype.str,
            order=order,
            fill_value=metadata.get("fill_value"),
            dimension_separator=separator,
            compressor=compressor,
            filters=filters,
        )

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype_string)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "pherc1447-raw-qc/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = fetch_bytes(url)
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError(f"metadata at {url} is not a JSON object")
    return parsed, {
        "url": url,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def cube_bounds(
    center_zyx: Sequence[int], radius: int, shape_zyx: Sequence[int]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    center = np.asarray(center_zyx, dtype=np.int64)
    shape = np.asarray(shape_zyx, dtype=np.int64)
    if center.shape != (3,) or shape.shape != (3,):
        raise ValueError("center and shape must contain z/y/x")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    lower = center - int(radius)
    upper = center + int(radius) + 1
    if np.any(lower < 0) or np.any(upper > shape):
        raise ValueError("requested cube is outside the raw volume")
    return lower, upper


def intersecting_chunk_indices(
    lower_zyx: Sequence[int],
    upper_zyx: Sequence[int],
    chunks_zyx: Sequence[int],
) -> tuple[tuple[int, int, int], ...]:
    lower = np.asarray(lower_zyx, dtype=np.int64)
    upper = np.asarray(upper_zyx, dtype=np.int64)
    chunks = np.asarray(chunks_zyx, dtype=np.int64)
    if np.any(lower < 0) or np.any(upper <= lower) or np.any(chunks <= 0):
        raise ValueError("invalid ROI or chunk bounds")
    first = lower // chunks
    last = (upper - 1) // chunks
    return tuple(
        tuple(int(value) for value in index)
        for index in itertools.product(
            range(int(first[0]), int(last[0]) + 1),
            range(int(first[1]), int(last[1]) + 1),
            range(int(first[2]), int(last[2]) + 1),
        )
    )


def chunk_shape(
    spec: ZarrV2ArraySpec, chunk_index_zyx: Sequence[int]
) -> tuple[int, int, int]:
    index = np.asarray(chunk_index_zyx, dtype=np.int64)
    chunks = np.asarray(spec.chunks_zyx, dtype=np.int64)
    shape = np.asarray(spec.shape_zyx, dtype=np.int64)
    start = index * chunks
    if np.any(index < 0) or np.any(start >= shape):
        raise ValueError("chunk index is outside the array")
    return tuple(int(value) for value in np.minimum(chunks, shape - start))


def decode_chunk_payload(
    payload: bytes,
    expected_shape: Sequence[int],
    spec: ZarrV2ArraySpec,
    *,
    blosc_path: Path = DEFAULT_BLOSC,
) -> tuple[NDArray[Any], str, bytes]:
    expected_count = int(np.prod(expected_shape, dtype=np.int64))
    expected_bytes = expected_count * spec.dtype.itemsize
    if spec.compressor is None:
        decoded = payload
        decoder = "zarr-v2-uncompressed"
    else:
        library = ctypes.CDLL(str(blosc_path.resolve()))
        library.blosc_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.blosc_decompress.restype = ctypes.c_int
        source = ctypes.create_string_buffer(payload)
        destination = ctypes.create_string_buffer(expected_bytes)
        decoded_size = int(
            library.blosc_decompress(source, destination, expected_bytes)
        )
        if decoded_size != expected_bytes:
            raise RuntimeError(
                f"expected {expected_bytes} decompressed bytes, got {decoded_size}"
            )
        decoded = destination.raw
        decoder = "vc3d-bundled-libblosc"
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"chunk payload has {len(decoded)} decoded bytes; expected {expected_bytes}"
        )
    array = np.frombuffer(decoded, dtype=spec.dtype).reshape(
        tuple(int(value) for value in expected_shape), order=spec.order
    )
    return array.copy(), decoder, decoded


ChunkLoader = Callable[
    [tuple[int, int, int], tuple[int, int, int]],
    tuple[NDArray[Any], Mapping[str, Any]],
]


def assemble_roi(
    spec: ZarrV2ArraySpec,
    lower_zyx: Sequence[int],
    upper_zyx: Sequence[int],
    loader: ChunkLoader,
) -> tuple[NDArray[Any], list[dict[str, Any]]]:
    lower = np.asarray(lower_zyx, dtype=np.int64)
    upper = np.asarray(upper_zyx, dtype=np.int64)
    shape = np.asarray(spec.shape_zyx, dtype=np.int64)
    chunks = np.asarray(spec.chunks_zyx, dtype=np.int64)
    if np.any(lower < 0) or np.any(upper <= lower) or np.any(upper > shape):
        raise ValueError("ROI is outside the array")
    output = np.full(
        tuple(int(value) for value in upper - lower),
        spec.fill_value if spec.fill_value is not None else 0,
        dtype=spec.dtype,
    )
    records: list[dict[str, Any]] = []
    for index in intersecting_chunk_indices(lower, upper, chunks):
        expected_shape = chunk_shape(spec, index)
        source, loader_record = loader(index, expected_shape)
        if tuple(source.shape) != expected_shape or source.dtype != spec.dtype:
            raise ValueError(
                f"loader returned {source.shape}/{source.dtype}; "
                f"expected {expected_shape}/{spec.dtype}"
            )
        source_start = np.asarray(index, dtype=np.int64) * chunks
        source_stop = source_start + np.asarray(expected_shape, dtype=np.int64)
        intersection_start = np.maximum(lower, source_start)
        intersection_stop = np.minimum(upper, source_stop)
        source_slices = tuple(
            slice(int(a), int(b))
            for a, b in zip(
                intersection_start - source_start, intersection_stop - source_start
            )
        )
        output_slices = tuple(
            slice(int(a), int(b))
            for a, b in zip(intersection_start - lower, intersection_stop - lower)
        )
        output[output_slices] = source[source_slices]
        records.append(
            {
                "chunk_index_zyx": list(index),
                "chunk_start_zyx": source_start.astype(int).tolist(),
                "chunk_shape_zyx": list(expected_shape),
                "intersection_start_zyx": intersection_start.astype(int).tolist(),
                "intersection_stop_zyx_exclusive": intersection_stop.astype(int).tolist(),
                **dict(loader_record),
            }
        )
    return output, records


def chunk_key(spec: ZarrV2ArraySpec, index_zyx: Sequence[int]) -> str:
    separator = spec.dimension_separator
    return separator.join(str(int(value)) for value in index_zyx)


def download_roi(
    root_url: str,
    array_path: str,
    spec: ZarrV2ArraySpec,
    lower_zyx: Sequence[int],
    upper_zyx: Sequence[int],
    *,
    blosc_path: Path = DEFAULT_BLOSC,
) -> tuple[NDArray[Any], list[dict[str, Any]]]:
    array_url = root_url.rstrip("/") + "/" + array_path.strip("/")

    def loader(
        index: tuple[int, int, int], expected_shape: tuple[int, int, int]
    ) -> tuple[NDArray[Any], Mapping[str, Any]]:
        url = array_url + "/" + chunk_key(spec, index)
        try:
            payload = fetch_bytes(url)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            fill = spec.fill_value if spec.fill_value is not None else 0
            array = np.full(expected_shape, fill, dtype=spec.dtype)
            decoded = array.tobytes(order=spec.order)
            return array, {
                "url": url,
                "status": "missing_chunk_fill_value",
                "downloaded_bytes": 0,
                "payload_sha256": None,
                "decoded_bytes": len(decoded),
                "decoded_sha256": sha256_bytes(decoded),
                "decoder": "zarr-v2-fill-value",
            }
        array, decoder, decoded = decode_chunk_payload(
            payload, expected_shape, spec, blosc_path=blosc_path
        )
        return array, {
            "url": url,
            "status": "downloaded",
            "downloaded_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "decoded_bytes": len(decoded),
            "decoded_sha256": sha256_bytes(decoded),
            "decoder": decoder,
        }

    return assemble_roi(spec, lower_zyx, upper_zyx, loader)


def orthogonal_slice(
    cube: NDArray[Any], plane: str, index: int
) -> NDArray[Any]:
    """Return a 2-D slice with labeled axes increasing right/down."""

    if cube.ndim != 3:
        raise ValueError("cube must have z/y/x dimensions")
    if plane == "xy":
        return cube[index, :, :]
    if plane == "xz":
        return cube[:, index, :]
    if plane == "yz":
        return cube[:, :, index]
    raise ValueError("plane must be xy, xz, or yz")


def display_window(
    cube: NDArray[Any], low_percentile: float = 1.0, high_percentile: float = 99.0
) -> tuple[float, float]:
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    finite = np.asarray(cube)[np.isfinite(cube)]
    if finite.size == 0:
        raise ValueError("cube contains no finite values")
    low, high = np.percentile(finite.astype(np.float64), [low_percentile, high_percentile])
    if not high > low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if not high > low:
        high = low + 1.0
    return float(low), float(high)


def window_uint8(array: NDArray[Any], low: float, high: float) -> NDArray[np.uint8]:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("display window must be finite and increasing")
    normalized = (np.asarray(array, dtype=np.float64) - low) * (255.0 / (high - low))
    return np.rint(np.clip(normalized, 0.0, 255.0)).astype(np.uint8)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_crosshair_and_normal(
    draw: ImageDraw.ImageDraw,
    origin_xy: tuple[int, int],
    panel_size: int,
    plane: str,
    normal_xyz: Sequence[float] | None,
) -> None:
    center = panel_size // 2
    ox, oy = origin_xy
    color = (0, 255, 255)
    draw.line((ox + center - 8, oy + center, ox + center + 8, oy + center), fill=color, width=1)
    draw.line((ox + center, oy + center - 8, ox + center, oy + center + 8), fill=color, width=1)
    if normal_xyz is None:
        return
    nx, ny, nz = (float(value) for value in normal_xyz)
    components = {"xy": (nx, ny), "xz": (nx, nz), "yz": (ny, nz)}[plane]
    magnitude = math.hypot(*components)
    if magnitude <= 1e-12:
        return
    dx, dy = (value / magnitude for value in components)
    length = panel_size * 0.28
    endpoint = (ox + center + dx * length, oy + center + dy * length)
    draw.line((ox + center, oy + center, *endpoint), fill=(255, 80, 180), width=2)
    angle = math.atan2(dy, dx)
    for delta in (-0.6, 0.6):
        back = angle + math.pi + delta
        draw.line(
            (
                endpoint[0],
                endpoint[1],
                endpoint[0] + 8 * math.cos(back),
                endpoint[1] + 8 * math.sin(back),
            ),
            fill=(255, 80, 180),
            width=2,
        )


def render_contact_sheet(
    cube: NDArray[Any],
    center_global_zyx: Sequence[int],
    offsets: Sequence[int],
    low: float,
    high: float,
    *,
    normal_xyz: Sequence[float] | None = None,
    scale: int = 4,
) -> Image.Image:
    radius_zyx = (np.asarray(cube.shape, dtype=np.int64) - 1) // 2
    if np.any(radius_zyx * 2 + 1 != np.asarray(cube.shape)):
        raise ValueError("contact sheet requires odd cube dimensions")
    if scale <= 0:
        raise ValueError("scale must be positive")
    panel = int(max(cube.shape) * scale)
    gap = 8
    label_height = 28
    left = 56
    top = 34
    width = left + len(offsets) * panel + max(0, len(offsets) - 1) * gap + 12
    height = top + 3 * (panel + label_height) + 2 * gap + 12
    canvas = Image.new("RGB", (width, height), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text(
        (left, 9),
        f"Raw CT orthogonal contact sheet | shared window {low:.1f}..{high:.1f} | no ink inference",
        fill=(235, 235, 235),
        font=font,
    )
    center_global = np.asarray(center_global_zyx, dtype=np.int64)
    plane_rows = (
        ("xy", "XY", 0, "z"),
        ("xz", "XZ", 1, "y"),
        ("yz", "YZ", 2, "x"),
    )
    for row, (plane, row_label, fixed_axis, fixed_name) in enumerate(plane_rows):
        y0 = top + row * (panel + label_height + gap)
        draw.text((10, y0 + panel // 2 - 6), row_label, fill=(220, 220, 220), font=font)
        for column, offset in enumerate(offsets):
            index = int(radius_zyx[fixed_axis] + int(offset))
            if not 0 <= index < cube.shape[fixed_axis]:
                raise ValueError(f"offset {offset} is outside cube for {plane}")
            source = window_uint8(orthogonal_slice(cube, plane, index), low, high)
            tile = Image.fromarray(source, mode="L").resize(
                (panel, panel), resample=Image.Resampling.NEAREST
            ).convert("RGB")
            x0 = left + column * (panel + gap)
            canvas.paste(tile, (x0, y0))
            coordinate = int(center_global[fixed_axis] + int(offset))
            draw.rectangle((x0, y0, x0 + panel - 1, y0 + panel - 1), outline=(100, 100, 105))
            draw.text(
                (x0, y0 + panel + 5),
                f"{fixed_name}={coordinate}  d={int(offset):+d}",
                fill=(225, 225, 225),
                font=font,
            )
            if int(offset) == 0:
                _draw_crosshair_and_normal(
                    draw, (x0, y0), panel, plane, normal_xyz
                )
    return canvas


def render_central_sheet(
    cube: NDArray[Any],
    center_global_zyx: Sequence[int],
    low: float,
    high: float,
    *,
    windowed: bool,
    normal_xyz: Sequence[float] | None = None,
    scale: int = 5,
) -> Image.Image:
    center_local = (np.asarray(cube.shape, dtype=np.int64) - 1) // 2
    panel = int(max(cube.shape) * scale)
    gap = 12
    left = 12
    top = 42
    label_height = 28
    canvas = Image.new(
        "RGB", (left * 2 + panel * 3 + gap * 2, top + panel + label_height + 12), (16, 16, 18)
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()
    mode_label = f"window {low:.1f}..{high:.1f}" if windowed else "native uint8 0..255"
    draw.text(
        (left, 10),
        f"Raw CT at selected seed | {mode_label} | cyan seed, magenta m7 PCA normal | no ink inference",
        fill=(235, 235, 235),
        font=font,
    )
    center_global = np.asarray(center_global_zyx, dtype=np.int64)
    entries = (
        ("xy", "XY", 0, "z"),
        ("xz", "XZ", 1, "y"),
        ("yz", "YZ", 2, "x"),
    )
    for column, (plane, label, axis, fixed_name) in enumerate(entries):
        source = orthogonal_slice(cube, plane, int(center_local[axis]))
        if windowed:
            source_u8 = window_uint8(source, low, high)
        elif np.asarray(source).dtype == np.uint8:
            source_u8 = np.asarray(source, dtype=np.uint8)
        else:
            source_u8 = window_uint8(source, 0.0, 255.0)
        tile = Image.fromarray(source_u8, mode="L").resize(
            (panel, panel), resample=Image.Resampling.NEAREST
        ).convert("RGB")
        x0 = left + column * (panel + gap)
        canvas.paste(tile, (x0, top))
        draw.rectangle((x0, top, x0 + panel - 1, top + panel - 1), outline=(100, 100, 105))
        draw.text(
            (x0, top + panel + 5),
            f"{label}  {fixed_name}={int(center_global[axis])}",
            fill=(225, 225, 225),
            font=font,
        )
        _draw_crosshair_and_normal(draw, (x0, top), panel, plane, normal_xyz)
    return canvas


def load_seed_normal(
    path: Path, expected_xyz: Sequence[int]
) -> tuple[tuple[float, float, float] | None, dict[str, Any] | None]:
    if not path.exists():
        return None, None
    payload = path.read_bytes()
    document = json.loads(payload)
    selected = document.get("selection", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError("seed evidence contains no selected candidate")
    if selected.get("global_xyz") != [int(value) for value in expected_xyz]:
        raise ValueError("seed evidence global_xyz does not match requested seed")
    normal = tuple(float(value) for value in selected.get("normal_xyz", ()))
    if len(normal) != 3 or not np.isfinite(normal).all():
        raise ValueError("seed evidence normal_xyz is invalid")
    norm = float(np.linalg.norm(normal))
    if norm <= 0:
        raise ValueError("seed evidence normal_xyz is zero")
    normal = tuple(float(value / norm) for value in normal)
    return normal, {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def write_outputs(
    output_dir: Path,
    cube: NDArray[Any],
    center_xyz: Sequence[int],
    radius: int,
    low: float,
    high: float,
    offsets: Sequence[int],
    *,
    normal_xyz: Sequence[float] | None,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    center_zyx = tuple(reversed(tuple(int(value) for value in center_xyz)))
    outputs: dict[str, dict[str, Any]] = {}

    cube_path = output_dir / f"raw-cube-r{radius}.npy"
    np.save(cube_path, cube, allow_pickle=False)
    outputs[cube_path.name] = {
        "bytes": cube_path.stat().st_size,
        "sha256": sha256_file(cube_path),
        "shape_zyx": list(cube.shape),
        "dtype": cube.dtype.str,
        "role": "lossless assembled raw CT cube",
    }

    center_local = (np.asarray(cube.shape, dtype=np.int64) - 1) // 2
    for plane, axis, fixed_name in (("xy", 0, "z"), ("xz", 1, "y"), ("yz", 2, "x")):
        source = orthogonal_slice(cube, plane, int(center_local[axis]))
        if source.dtype != np.uint8:
            raise ValueError("lossless PNG output currently requires uint8 raw CT")
        coordinate = int(center_zyx[axis])
        path = output_dir / f"raw-{plane}-{fixed_name}{coordinate}.png"
        Image.fromarray(source, mode="L").save(path, format="PNG", optimize=False)
        outputs[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "width": int(source.shape[1]),
            "height": int(source.shape[0]),
            "role": f"lossless native-intensity central {plane.upper()} slice",
        }

    rendered = {
        "orthogonal-central-native.png": render_central_sheet(
            cube, center_zyx, low, high, windowed=False, normal_xyz=normal_xyz
        ),
        "orthogonal-central-windowed.png": render_central_sheet(
            cube, center_zyx, low, high, windowed=True, normal_xyz=normal_xyz
        ),
        "orthogonal-contact-sheet-windowed.png": render_contact_sheet(
            cube, center_zyx, offsets, low, high, normal_xyz=normal_xyz
        ),
    }
    for filename, image in rendered.items():
        path = output_dir / filename
        image.save(path, format="PNG", optimize=False)
        outputs[filename] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "width": image.width,
            "height": image.height,
            "role": "annotated raw CT visibility aid",
        }
    return outputs


def _array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=DEFAULT_XYZ[0])
    parser.add_argument("--y", type=int, default=DEFAULT_XYZ[1])
    parser.add_argument("--z", type=int, default=DEFAULT_XYZ[2])
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--array-path", default=DEFAULT_ARRAY_PATH)
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-evidence", type=Path, default=DEFAULT_SEED_EVIDENCE)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    args = parser.parse_args()

    if not math.isfinite(args.voxel_um) or args.voxel_um <= 0:
        raise ValueError("--voxel-um must be finite and positive")

    root = args.raw_root.rstrip("/")
    array_path = args.array_path.strip("/")
    group_metadata, group_provenance = fetch_json(root + "/.zgroup")
    if group_metadata.get("zarr_format") != 2:
        raise ValueError("raw root is not a Zarr-v2 group")
    array_metadata, array_provenance = fetch_json(
        root + "/" + array_path + "/.zarray"
    )
    root_attributes, attributes_provenance = fetch_json(root + "/.zattrs")
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)

    center_xyz = (int(args.x), int(args.y), int(args.z))
    center_zyx = tuple(reversed(center_xyz))
    lower, upper = cube_bounds(center_zyx, int(args.radius), spec.shape_zyx)
    cube, chunk_records = download_roi(
        root,
        array_path,
        spec,
        lower,
        upper,
        blosc_path=args.blosc,
    )
    expected_shape = (2 * int(args.radius) + 1,) * 3
    if tuple(cube.shape) != expected_shape:
        raise RuntimeError(f"assembled cube is {cube.shape}; expected {expected_shape}")
    normal_xyz, evidence_provenance = load_seed_normal(
        args.seed_evidence, center_xyz
    )
    low, high = display_window(cube, args.low_percentile, args.high_percentile)
    outputs = write_outputs(
        args.output,
        cube,
        center_xyz,
        int(args.radius),
        low,
        high,
        DEFAULT_OFFSETS,
        normal_xyz=normal_xyz,
    )
    unique_values, counts = np.unique(cube, return_counts=True)
    histogram = {
        str(int(value)): int(count) for value, count in zip(unique_values, counts)
    }
    manifest = {
        "schema_version": 1,
        "purpose": "raw CT single-sheet visibility aids around a preselected m7 seed",
        "ink_inference_performed": False,
        "coordinate_convention": {
            "input": "xyz voxel indices",
            "array": "zyx voxel indices",
            "slice_axes": {
                "xy": "x increases right; y increases down",
                "xz": "x increases right; z increases down",
                "yz": "y increases right; z increases down",
            },
        },
        "seed_xyz": list(center_xyz),
        "seed_zyx": list(center_zyx),
        "seed_m7_normal_xyz": list(normal_xyz) if normal_xyz is not None else None,
        "seed_evidence": evidence_provenance,
        "voxel_size_um": float(args.voxel_um),
        "cube": {
            "radius_voxels": int(args.radius),
            "lower_zyx_inclusive": lower.astype(int).tolist(),
            "upper_zyx_exclusive": upper.astype(int).tolist(),
            "shape_zyx": list(cube.shape),
            "physical_side_length_mm": [
                float(length * args.voxel_um / 1000.0) for length in cube.shape
            ],
            "dtype": cube.dtype.str,
            "minimum": int(np.min(cube)),
            "maximum": int(np.max(cube)),
            "mean": float(np.mean(cube, dtype=np.float64)),
            "standard_deviation": float(np.std(cube, dtype=np.float64)),
            "histogram_uint8": histogram,
            "content_sha256": _array_sha256(cube),
        },
        "display": {
            "mapping": "global linear percentile window; clipped; no local enhancement",
            "low_percentile": float(args.low_percentile),
            "high_percentile": float(args.high_percentile),
            "low_value": low,
            "high_value": high,
            "contact_sheet_offsets_voxels": list(DEFAULT_OFFSETS),
            "annotations": {
                "cyan": "selected seed projection",
                "magenta": "projected m7 PCA normal when seed evidence is present",
            },
        },
        "source": {
            "root_url": root,
            "array_path": array_path,
            "zarr_group_metadata": group_provenance,
            "zarr_array_metadata": array_provenance,
            "zarr_root_attributes": attributes_provenance,
            "array_spec": asdict(spec),
            "root_multiscales": root_attributes.get("multiscales"),
            "chunks_intersecting_roi": len(chunk_records),
            "downloaded_bytes": int(
                sum(int(record["downloaded_bytes"]) for record in chunk_records)
            ),
            "chunks": chunk_records,
            "blosc_fallback": {
                "used": spec.compressor is not None,
                "path": str(args.blosc.resolve()),
                "sha256": sha256_file(args.blosc),
            },
        },
        "outputs": outputs,
        "limitations": [
            "The cube covers only a 0.562 mm neighborhood and cannot validate a grown surface.",
            "Orthogonal slices can hide ambiguity aligned obliquely to all three views.",
            "The contrast-windowed PNGs are display aids; lossless source values remain in the NPY and central raw PNGs.",
            "No ink model, letter detector, or semantic interpretation was applied.",
        ],
    }
    manifest_path = args.output / "raw-ct-cube-manifest.json"
    rendered = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
