#!/usr/bin/env python3
"""Freeze an overlap-component-isolated reviewed-patch benchmark split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor.audit import AuditConfig  # noqa: E402
from tifxyz_doctor.reviewed_benchmark import overlap_isolated_split  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "reviewed-same-wrap-split-v1.json"
DETECTOR_COMMIT = "d3c8309ca707e2f18e7d64e38fb7be4ff4ca77c0"
SOURCE_BUCKET = (
    "hf://buckets/scrollprize/datasets/"
    "spiral/PHercParis4/verified_patches"
)
OVERLAP_BUCKET_PATH = (
    "hf://buckets/scrollprize/datasets/"
    "spiral/PHercParis4/patch-overlap-pcls.json"
)
OVERLAP_URL = (
    "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
    "spiral/PHercParis4/patch-overlap-pcls.json"
)
OVERLAP_SHA256 = (
    "11fc0ef6112a2b9829f80242b7c67530"
    "c841b1159aed1ed5bb45e4362aad5097"
)
REQUIRED_FILES = (
    "meta.json",
    "corr_points_results.json",
    "x.tif",
    "y.tif",
    "z.tif",
)
DEVELOPMENT_LIMIT = 64
HOLDOUT_TARGET = 128
HOLDOUT_SALT = "tifxyz-doctor-reviewed-holdout-v1"
OFFSETS_VOXELS = (4.0, 8.0, 16.0)
TRANSITION_WIDTHS_CELLS = (1, 4, 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _select_evenly(ids: list[str], limit: int) -> set[str]:
    if limit >= len(ids):
        return set(ids)
    indices = np.linspace(0, len(ids) - 1, num=limit, dtype=np.int64)
    return {ids[int(index)] for index in indices}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--overlap-graph",
        type=Path,
        help="Defaults to DATA/patch-overlap-pcls.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overlap_path = args.overlap_graph or args.data / "patch-overlap-pcls.json"
    if not overlap_path.is_file():
        raise SystemExit(f"overlap graph not found: {overlap_path}")
    overlap_sha256 = _sha256(overlap_path)
    if overlap_sha256 != OVERLAP_SHA256:
        raise SystemExit(
            f"overlap graph SHA-256 mismatch: {overlap_sha256} "
            f"(expected {OVERLAP_SHA256})"
        )
    candidate_ids = [
        path.name
        for path in sorted(args.data.glob("same_wrap*"))
        if path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)
    ]
    if not candidate_ids:
        raise SystemExit("no complete reviewed same_wrap patches found")
    development_ids = _select_evenly(candidate_ids, DEVELOPMENT_LIMIT)
    with overlap_path.open("r", encoding="utf-8") as handle:
        overlap_graph = json.load(handle)
    split = overlap_isolated_split(
        overlap_graph,
        set(candidate_ids),
        development_ids,
        holdout_target_patches=HOLDOUT_TARGET,
        salt=HOLDOUT_SALT,
    )
    component_count = len(split["components"])
    contaminated_components = {
        split["component_ids"][patch_id]
        for patch_id in split["development_connected_ids"]
    }
    clean_components = {
        split["component_ids"][patch_id]
        for patch_id in split["clean_holdout_pool_ids"]
    }
    selected_components = {
        split["component_ids"][patch_id]
        for patch_id in split["selected_holdout_ids"]
    }
    result = {
        "schema_version": "reviewed-same-wrap-split-v1",
        "source": {
            "verified_patch_bucket": SOURCE_BUCKET,
            "selected_patch_count": len(candidate_ids),
            "selected_patch_ids_sha256": _ids_sha256(candidate_ids),
            "required_files": list(REQUIRED_FILES),
            "overlap_graph": {
                "bucket_path": OVERLAP_BUCKET_PATH,
                "url": OVERLAP_URL,
                "sha256": overlap_sha256,
            },
            "license": {
                "spdx": "CC-BY-NC-4.0",
                "url": "https://creativecommons.org/licenses/by-nc/4.0/",
            },
        },
        "frozen_detector": {
            "commit": DETECTOR_COMMIT,
            "configuration": asdict(AuditConfig()),
        },
        "protocol": {
            "development_selection": (
                "The exact 64 evenly spaced patch IDs used while developing "
                "the coherent-normal-step cue before the detector commit."
            ),
            "holdout_isolation": (
                "Every patch in any overlap-connected component touching a "
                "development patch is excluded from holdout."
            ),
            "holdout_component_selection": (
                "Rank clean components by SHA-256 of a fixed salt and their "
                "sorted IDs; include whole components until at least 128 "
                "patches are selected."
            ),
            "holdout_salt": HOLDOUT_SALT,
            "holdout_target_patches": HOLDOUT_TARGET,
            "normal_offsets_voxels": list(OFFSETS_VOXELS),
            "transition_widths_cells": list(TRANSITION_WIDTHS_CELLS),
            "annotation_neighborhood_radius_cells": 4,
            "synthetic_evaluation_radius_cells": 1,
            "primary_metrics": [
                "Exact null: coordinate bytes, validity bytes, public report, and audit signature.",
                "Incremental event detection within one cell of the injected seam.",
                "Same-wrap annotation-neighborhood alert rates.",
                "Overlap-component cluster-bootstrap intervals for patch-level rates.",
            ],
        },
        "counts": {
            "overlap_edges": len(split["overlap_edges"]),
            "overlap_components": component_count,
            "development_patches": len(split["development_ids"]),
            "development_connected_components": len(contaminated_components),
            "development_connected_patches": len(
                split["development_connected_ids"]
            ),
            "development_related_excluded_patches": len(
                split["development_related_excluded_ids"]
            ),
            "clean_holdout_components": len(clean_components),
            "clean_holdout_pool_patches": len(split["clean_holdout_pool_ids"]),
            "selected_holdout_components": len(selected_components),
            "selected_holdout_patches": len(split["selected_holdout_ids"]),
        },
        "split": split,
    }
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({len(candidate_ids)} patches, "
        f"{len(split['development_connected_ids'])} development-connected, "
        f"{len(split['clean_holdout_pool_ids'])} clean holdout pool, "
        f"{len(split['selected_holdout_ids'])} selected synthetic holdout)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
