#!/usr/bin/env python3
"""Freeze model-blind compressed-region panels from held-out GT geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


MAX_SITES = 30_000
SPACING_CAP = 40
COMPRESSED_MAX = 8.0
PANEL_COUNT = 4


def geometry_normals_at(gt: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return unit SDF-gradient normals at sampled points in ZYX order.

    Gradient axes are materialized one at a time. This is algebraically the
    same sigma-3 signed-distance normal as a full 3xZYX stack, but avoids a
    several-hundred-megabyte peak on the official 256-cube labels.
    """
    sdf = ndi.distance_transform_edt(~gt).astype(np.float32)
    sdf -= ndi.distance_transform_edt(gt).astype(np.float32)
    ndi.gaussian_filter(sdf, 3.0, output=sdf)
    pz, py, px = points.T
    sampled = np.empty((len(points), 3), dtype=np.float32)
    for axis in range(3):
        component = np.gradient(sdf, axis=axis).astype(np.float32, copy=False)
        sampled[:, axis] = component[pz, py, px]
        del component
    magnitude = np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-6
    return sampled / magnitude


def spacing_at(points: np.ndarray, gt: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Measure the next sheet along either local-normal direction."""
    shape = np.asarray(gt.shape)
    spacing = np.full(len(points), float(SPACING_CAP), dtype=np.float32)
    if normals.shape != points.shape:
        raise ValueError("normals must be sampled Nx3 vectors")
    pz, py, px = points.T
    nz, ny, nx = normals.T
    for sign in (1.0, -1.0):
        left_own = np.zeros(len(points), dtype=bool)
        found = np.zeros(len(points), dtype=bool)
        for distance in range(1, SPACING_CAP):
            sample = np.column_stack(
                (
                    pz + sign * distance * nz,
                    py + sign * distance * ny,
                    px + sign * distance * nx,
                )
            )
            sample = np.rint(sample).astype(np.int64)
            sample = np.clip(sample, 0, shape - 1)
            hit = gt[sample[:, 0], sample[:, 1], sample[:, 2]]
            left_own |= ~hit
            newly_found = left_own & hit & ~found
            spacing[newly_found] = np.minimum(spacing[newly_found], distance)
            found |= newly_found
            if found.all():
                break
    return spacing


def deterministic_sites(gt: np.ndarray, filename: str) -> np.ndarray:
    sites = np.argwhere(gt)
    if len(sites) < 5_000:
        raise ValueError(f"{filename}: fewer than 5,000 labelled surface voxels")
    seed = int.from_bytes(hashlib.sha256(filename.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(sites), size=min(MAX_SITES, len(sites)), replace=False)
    return sites[chosen]


def representative_center(points: np.ndarray, spacing: np.ndarray) -> list[int]:
    compressed = points[spacing <= COMPRESSED_MAX]
    if not len(compressed):
        compressed = points[spacing == spacing.min()]
    median = np.median(compressed, axis=0)
    distance = np.square(compressed - median).sum(axis=1)
    best_distance = distance.min()
    tied = compressed[distance == best_distance]
    order = np.lexsort((tied[:, 2], tied[:, 1], tied[:, 0]))
    return [int(value) for value in tied[order[0]]]


def rank_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda record: (
            -record["compressed_fraction_le_8"],
            record["median_spacing"],
            record["image"],
        ),
    )


def measure_candidate(root: Path, record: dict) -> dict:
    labels = {
        path.name: path
        for path in root.rglob("*.tif")
        if not path.name.endswith("_0000.tif")
    }
    label = labels.get(record["label"])
    if label is None:
        raise RuntimeError(f"missing candidate label for {record['image']}")
    gt = tifffile.imread(label) > 0
    sites = deterministic_sites(gt, record["image"])
    spacing = spacing_at(sites, gt, geometry_normals_at(gt, sites))
    compressed_fraction = float(np.mean(spacing <= COMPRESSED_MAX))
    return {
        **record,
        "sampled_surface_sites": int(len(sites)),
        "compressed_sites_le_8": int(np.sum(spacing <= COMPRESSED_MAX)),
        "compressed_fraction_le_8": compressed_fraction,
        "median_spacing": float(np.median(spacing)),
        "spacing_p10": float(np.percentile(spacing, 10)),
        "center_zyx": representative_center(sites, spacing),
    }


def measure_candidate_isolated(root: Path, record: dict) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--root",
        str(root),
        "--measure-record",
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"panel measurement failed for {record['image']}: {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid panel worker output for {record['image']}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--measure-record")
    args = parser.parse_args()
    if args.measure_record is not None:
        record = json.loads(args.measure_record)
        print(json.dumps(measure_candidate(args.root, record), sort_keys=True))
        return 0
    if args.split_manifest is None or args.out is None:
        parser.error("--split-manifest and --out are required outside worker mode")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    candidates = [record for record in split["records"] if record["split"] == "test"]
    if len(candidates) < PANEL_COUNT:
        raise RuntimeError("real test split has too few panel candidates")

    measured = []
    for index, record in enumerate(candidates, start=1):
        panel = measure_candidate_isolated(args.root, record)
        measured.append(panel)
        print(
            f"PANEL_SCAN {index}/{len(candidates)} {record['image']} "
            f"compressed={panel['compressed_fraction_le_8']:.6f}",
            flush=True,
        )

    ranked = rank_records(measured)
    selected = ranked[:PANEL_COUNT]
    if any(record["compressed_sites_le_8"] == 0 for record in selected):
        raise RuntimeError("fewer than four test patches contain sampled <=8-voxel sites")
    payload = {
        "schema_version": "1.0",
        "status": "model-blind; selected from held-out labels before prediction",
        "rule": (
            "all s4/s5 test labels with >=5000 foreground voxels; sample up to "
            "30000 foreground sites using SHA256(filename); rank by descending "
            "share with next-sheet normal spacing <=8 voxels, then ascending "
            "median spacing and filename; take first four"
        ),
        "spacing": {
            "normal": "gradient of sigma-3 signed-distance field",
            "ray_directions": [1, -1],
            "cap_voxels": SPACING_CAP,
            "compressed_max_voxels": COMPRESSED_MAX,
        },
        "source_split_records_sha256": split["records_sha256"],
        "candidate_count": len(measured),
        "selected": selected,
        "all_candidate_measurements": ranked,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("selected:", [record["image"] for record in payload["selected"]])
    print("payload SHA-256:", payload["payload_sha256"])
    print("REAL_PANELS_FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
