#!/usr/bin/env python3
"""Sample one PHerc1447 m7 chunk with the VC3D-bundled Blosc codec.

This is intentionally small and dependency-light: public HTTP, one Zarr-v2
chunk, and NumPy.  It reports the requested voxel plus the deterministic
nearest foreground voxel inside a bounded same-chunk search cube.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np


CHUNK_EDGE = 192
DEFAULT_PREFIX = (
    "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1447/"
    "representations/predictions/surfaces/"
    "20250521151220-surface-20260413222639-surface-m7-L0-th0.2.zarr/0"
)
DEFAULT_BLOSC = Path(
    "work/first-letters/vc3d-stable/unpacked/VC3D.app/Contents/Frameworks/"
    "libblosc.1.dylib"
)


def contiguous_runs(line: np.ndarray) -> list[list[int]]:
    padded = np.pad(np.asarray(line, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [[int(left), int(right)] for left, right in zip(starts, stops)]


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
    zyx = xyz[::-1]
    chunk_zyx = zyx // CHUNK_EDGE
    local_zyx = zyx % CHUNK_EDGE
    chunk_url = args.prefix.rstrip("/") + "/" + "/".join(map(str, chunk_zyx))
    with urllib.request.urlopen(chunk_url, timeout=60) as response:
        compressed = response.read()

    library = ctypes.CDLL(str(args.blosc.resolve()))
    library.blosc_decompress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.blosc_decompress.restype = ctypes.c_int
    output_size = CHUNK_EDGE**3
    source = ctypes.create_string_buffer(compressed)
    destination = ctypes.create_string_buffer(output_size)
    decompressed = int(
        library.blosc_decompress(source, destination, output_size)
    )
    if decompressed != output_size:
        raise RuntimeError(
            f"expected {output_size} decompressed bytes, got {decompressed}"
        )
    chunk = np.frombuffer(destination, dtype=np.uint8).reshape(
        (CHUNK_EDGE,) * 3
    )

    radius = int(args.radius)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    lower = local_zyx - radius
    upper = local_zyx + radius + 1
    if np.any(lower < 0) or np.any(upper > CHUNK_EDGE):
        raise ValueError("bounded search crosses a chunk edge; fetch more chunks")
    cube = chunk[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))]
    foreground_local_cube = np.argwhere(cube == 255)
    nearest: dict[str, object] | None = None
    if foreground_local_cube.size:
        offsets = foreground_local_cube - radius
        squared = np.sum(offsets * offsets, axis=1)
        # Stable row-major argmin supplies a deterministic z,y,x tie break.
        index = int(np.argmin(squared))
        nearest_local_zyx = lower + foreground_local_cube[index]
        nearest_global_zyx = chunk_zyx * CHUNK_EDGE + nearest_local_zyx
        nz, ny, nx = map(int, nearest_local_zyx)
        nearest = {
            "global_xyz": [
                int(nearest_global_zyx[2]),
                int(nearest_global_zyx[1]),
                int(nearest_global_zyx[0]),
            ],
            "global_zyx": [int(v) for v in nearest_global_zyx],
            "offset_xyz": [int(v) for v in offsets[index][::-1]],
            "distance_voxels": float(np.sqrt(squared[index])),
            "value": int(chunk[nz, ny, nx]),
            "axis_runs_through_candidate": {
                "z": contiguous_runs(chunk[:, ny, nx] == 255),
                "y": contiguous_runs(chunk[nz, :, nx] == 255),
                "x": contiguous_runs(chunk[nz, ny, :] == 255),
            },
        }

    lz, ly, lx = map(int, local_zyx)
    result = {
        "array_axis_order": "zyx",
        "requested_xyz": [int(v) for v in xyz],
        "requested_zyx": [int(v) for v in zyx],
        "requested_value": int(chunk[lz, ly, lx]),
        "required_value": 255,
        "requested_pass": int(chunk[lz, ly, lx]) == 255,
        "search_radius_voxels": radius,
        "foreground_voxels_in_search_cube": int(np.count_nonzero(cube == 255)),
        "nearest_foreground": nearest,
        "chunk_index_zyx": [int(v) for v in chunk_zyx],
        "local_index_zyx": [int(v) for v in local_zyx],
        "chunk_url": chunk_url,
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "decompressed_bytes": decompressed,
        "decompressed_sha256": hashlib.sha256(destination.raw).hexdigest(),
        "codec_path": str(args.blosc.resolve()),
        "codec_sha256": hashlib.sha256(args.blosc.read_bytes()).hexdigest(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if nearest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
