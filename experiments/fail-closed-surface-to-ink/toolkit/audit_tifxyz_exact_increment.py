#!/usr/bin/env python3
"""Measure byte-exact shared and newly added physical area between TIFFXYZ masks.

The audit is deliberately conservative: a quad is shared only when all four
float32 XYZ vertices match byte-for-byte, independent of raster location or
vertex order.  Everything else is reported as new/lost rather than inferred
to be equivalent by a distance tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile

from sample_raw_ct_seed_cube import sha256_file
from tifxyz_render_pipeline import load_tifxyz_asset


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "tifxyz-byte-exact-quad-increment-v1"


def _quad_table(directory: Path, voxel_um: float) -> tuple[dict[tuple[bytes, ...], float], dict]:
    asset = load_tifxyz_asset(directory, resolution="stored")
    points = np.asarray(asset.surface.points_xyz, dtype=np.float32)
    mask = np.asarray(asset.surface.valid, dtype=bool)
    quads = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    table: dict[tuple[bytes, ...], float] = {}
    for row, column in np.argwhere(quads):
        vertices = np.asarray(
            [
                points[row, column],
                points[row, column + 1],
                points[row + 1, column],
                points[row + 1, column + 1],
            ],
            dtype=np.float32,
        )
        key = tuple(sorted(vertex.tobytes() for vertex in vertices))
        if key in table:
            raise ValueError("duplicate canonical quad geometry within one surface")
        p00, p01, p10, p11 = vertices.astype(np.float64)
        area_vox2 = 0.5 * (
            np.linalg.norm(np.cross(p01 - p00, p10 - p00))
            + np.linalg.norm(np.cross(p11 - p10, p11 - p01))
        )
        table[key] = float(area_vox2 * voxel_um * voxel_um / 100_000_000.0)
    return table, {
        "directory": str(directory.resolve()),
        "input_sha256": dict(asset.input_sha256),
        "stored_shape": list(mask.shape),
        "valid_vertex_count": int(np.count_nonzero(mask)),
        "active_quad_count": len(table),
        "area_cm2": float(sum(table.values())),
    }


def audit(old: Path, new: Path, voxel_um: float) -> dict:
    old_table, old_input = _quad_table(old, voxel_um)
    new_table, new_input = _quad_table(new, voxel_um)
    old_keys, new_keys = set(old_table), set(new_table)
    shared = old_keys & new_keys
    new_only = new_keys - old_keys
    old_only = old_keys - new_keys
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "comparison_semantics": (
            "a shared quad requires byte-identical float32 XYZ at all four vertices; "
            "raster location and vertex order do not matter; no distance tolerance"
        ),
        "voxel_size_um": float(voxel_um),
        "inputs": {"old": old_input, "new": new_input},
        "result": {
            "shared_exact_quad_count": len(shared),
            "new_only_exact_quad_count": len(new_only),
            "old_only_exact_quad_count": len(old_only),
            "shared_area_cm2_measured_on_new": float(sum(new_table[key] for key in shared)),
            "new_only_area_cm2": float(sum(new_table[key] for key in new_only)),
            "old_only_area_cm2": float(sum(old_table[key] for key in old_only)),
            "new_area_fraction_byte_exactly_shared": (
                float(sum(new_table[key] for key in shared) / sum(new_table.values()))
                if new_table
                else 0.0
            ),
        },
        "limitations": [
            "Non-identical quads may still sample overlapping physical surface; this audit never labels them shared.",
            "This is an exact provenance/novelty audit, not a closest-surface or normal-slab overlap proof.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-um", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit(args.old, args.new, args.voxel_um)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), **report["result"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
