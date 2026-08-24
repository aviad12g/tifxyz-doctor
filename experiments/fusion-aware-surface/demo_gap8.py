#!/usr/bin/env python3
"""Render a deterministic, checkpoint-free demonstration of Gap8 supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gap_supervision import inter_sheet_gap_mask


SIZE = 96
RADIUS = 8
GAP_WEIGHT = 8
SCALE = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def toy_instances() -> np.ndarray:
    """Return two nearby, gently curved sheet instances on a public toy grid."""
    labels = np.zeros((SIZE, SIZE), dtype=np.int16)
    for y in range(8, SIZE - 8):
        phase = 2.5 * np.sin((y - 8) * np.pi / 30.0)
        left = int(round(35 + phase))
        right = int(round(49 + phase))
        labels[y, left - 1 : left + 2] = 1
        labels[y, right - 1 : right + 2] = 2
    return labels


def render(labels: np.ndarray, gap: np.ndarray) -> Image.Image:
    occupied = labels > 0

    instances = np.full((SIZE, SIZE, 3), 18, dtype=np.uint8)
    instances[labels == 1] = (48, 191, 255)
    instances[labels == 2] = (255, 150, 52)

    gap_panel = np.full((SIZE, SIZE, 3), 18, dtype=np.uint8)
    gap_panel[occupied] = (130, 130, 130)
    gap_panel[gap] = (255, 52, 170)

    weights = np.full((SIZE, SIZE, 3), 32, dtype=np.uint8)
    weights[occupied] = (96, 96, 96)
    weights[gap] = (255, 255, 255)

    separator = np.full((SIZE, 4, 3), 235, dtype=np.uint8)
    canvas = np.concatenate(
        [instances, separator, gap_panel, separator, weights], axis=1
    )
    return Image.fromarray(canvas, mode="RGB").resize(
        (canvas.shape[1] * SCALE, canvas.shape[0] * SCALE),
        resample=Image.Resampling.NEAREST,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("gap8-demo-output"))
    args = parser.parse_args()

    labels = toy_instances()
    gap = inter_sheet_gap_mask(labels, surface=labels > 0, radius=RADIUS)
    if np.any(gap & (labels > 0)):
        raise RuntimeError("demo gap mask intersects foreground")
    if not np.any(gap):
        raise RuntimeError("demo geometry produced no inter-sheet gap")

    args.out.mkdir(parents=True, exist_ok=True)
    image_path = args.out / "gap8_demo.png"
    summary_path = args.out / "demo_summary.json"
    manifest_path = args.out / "demo_manifest.json"

    render(labels, gap).save(image_path, format="PNG", optimize=False)

    voxel_count = int(labels.size)
    gap_voxels = int(gap.sum())
    control_total_weight = voxel_count
    gap8_total_weight = voxel_count + (GAP_WEIGHT - 1) * gap_voxels
    normalized_gap_gradient_multiplier = (
        GAP_WEIGHT * control_total_weight / gap8_total_weight
    )
    summary = {
        "demo_scope": "public toy geometry; no model checkpoint or result data",
        "gap_radius_voxels": RADIUS,
        "gap_voxels": gap_voxels,
        "gap_weight": GAP_WEIGHT,
        "image_panels_left_to_right": [
            "two exact sheet instances",
            "background reached by both instances",
            "resulting loss-weight map",
        ],
        "normalized_gap_gradient_multiplier_vs_control": round(
            normalized_gap_gradient_multiplier, 9
        ),
        "sheet_1_voxels": int((labels == 1).sum()),
        "sheet_2_voxels": int((labels == 2).sum()),
        "shape_yx": [SIZE, SIZE],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "files": {
            image_path.name: {"sha256": sha256(image_path), "bytes": image_path.stat().st_size},
            summary_path.name: {
                "sha256": sha256(summary_path),
                "bytes": summary_path.stat().st_size,
            },
        },
        "schema_version": "1.0",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Gap8 mechanism demo written to {args.out}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
