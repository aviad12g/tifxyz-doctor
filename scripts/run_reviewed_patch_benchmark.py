#!/usr/bin/env python3
"""Benchmark Doctor on reviewed same-wrap corridors and controlled switch proxies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor.audit import AuditConfig, audit_mesh  # noqa: E402
from tifxyz_doctor.io import load_tifxyz  # noqa: E402
from tifxyz_doctor.reviewed_benchmark import (  # noqa: E402
    inject_normal_offset_switch,
    same_wrap_corridor,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "reviewed-same-wrap-results-v0.2.0.json"
DEFAULT_OFFSETS_VOXELS = (4.0, 8.0, 16.0)
DEFAULT_TRANSITION_WIDTHS = (1, 4, 12)
DEFAULT_CORRIDOR_RADIUS = 4
DEFAULT_SYNTHETIC_LIMIT = 64
REQUIRED_FILES = (
    "meta.json",
    "corr_points_results.json",
    "x.tif",
    "y.tif",
    "z.tif",
)
SOURCE_BUCKET = (
    "hf://buckets/scrollprize/datasets/"
    "spiral/PHercParis4/verified_patches"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Root containing downloaded same_wrap* patches")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Sequential retries for transient file-read failures",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--synthetic-limit", type=int, default=DEFAULT_SYNTHETIC_LIMIT)
    parser.add_argument("--corridor-radius", type=int, default=DEFAULT_CORRIDOR_RADIUS)
    parser.add_argument(
        "--offsets",
        default=",".join(str(value) for value in DEFAULT_OFFSETS_VOXELS),
        help="Comma-separated normal offsets in voxels",
    )
    parser.add_argument(
        "--transition-widths",
        default=",".join(str(value) for value in DEFAULT_TRANSITION_WIDTHS),
        help="Comma-separated transition widths in grid cells",
    )
    return parser


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("offsets must contain positive numbers")
    return values


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 1 for item in values):
        raise ValueError("transition widths must contain positive integers")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_signature(report: dict[str, Any]) -> dict[str, Any]:
    cue_mask = np.asarray(report["_arrays"]["review_cue_mask"], dtype=bool)
    cue_digest = hashlib.sha256(np.packbits(cue_mask).tobytes()).hexdigest()
    geometry = report["geometry"]
    return {
        "finding_codes": [finding["code"] for finding in report["findings"]],
        "cue_mask_shape": list(cue_mask.shape),
        "cue_mask_sha256": cue_digest,
        "valid_quad_count": report["topology"]["valid_quad_count"],
        "long_edges": geometry["long_edges"],
        "short_edges": geometry["short_edges"],
        "degenerate_triangles": geometry["degenerate_triangles"],
        "folded_quads": geometry["folded_quads"],
        "normal_jumps": geometry["normal_jumps"]["jumps_above_threshold"],
        "coherent_normal_step_components": geometry["coherent_normal_steps"][
            "coherent_components"
        ],
        "coherent_normal_step_cells": geometry["coherent_normal_steps"][
            "coherent_cells"
        ],
        "high_condition_cells": geometry["high_condition_cells"],
        "high_symmetric_stretch_cells": geometry["high_symmetric_stretch_cells"],
        "high_symmetric_dirichlet_cells": geometry[
            "high_symmetric_dirichlet_cells"
        ],
        "high_area_distortion_cells": geometry["high_area_distortion_cells"],
        "high_shear_cells": geometry["high_shear_cells"],
        "nonlocal_proximity_pairs": report["nonlocal_proximity"]["pair_count"],
    }


def _same_wrap_point_count(results: dict[str, Any]) -> int:
    total = 0
    for point in results.get("points_list", []):
        if not isinstance(point, dict) or point.get("valid") is False:
            continue
        locations = point.get("model_locations", [])
        if isinstance(locations, list) and any(
            isinstance(location, dict) and "h" in location and "w" in location
            for location in locations
        ):
            total += 1
    return total


def _base_observation(
    path: Path,
    config: AuditConfig,
    corridor_radius: int,
) -> tuple[dict[str, Any], Any, dict[str, Any], np.ndarray]:
    with (path / "meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with (path / "corr_points_results.json").open("r", encoding="utf-8") as handle:
        corr_results = json.load(handle)
    data = load_tifxyz(path)
    report = audit_mesh(data, config)
    cue_mask = np.asarray(report["_arrays"]["review_cue_mask"], dtype=bool)
    normal_step_mask = np.asarray(
        report["_arrays"]["coherent_normal_step_cells"],
        dtype=bool,
    )
    valid_cells = np.asarray(report["_arrays"]["valid_cells"], dtype=bool)
    corridor = same_wrap_corridor(
        corr_results,
        data.shape,
        radius=corridor_radius,
    )
    labeled_cells = corridor & valid_cells
    corridor_cues = cue_mask & labeled_cells
    corridor_normal_steps = normal_step_mask & labeled_cells
    legacy_cues = cue_mask & ~normal_step_mask
    corridor_legacy_cues = legacy_cues & labeled_cells
    file_hashes = {name: _sha256(path / name) for name in REQUIRED_FILES}
    finding_codes = [finding["code"] for finding in report["findings"]]
    observation = {
        "id": path.name,
        "uuid": metadata.get("uuid"),
        "reviewed": metadata.get("tags", {}).get("reviewed"),
        "shape_hw": list(data.shape),
        "same_wrap_annotation_points": _same_wrap_point_count(corr_results),
        "valid_vertices": report["integrity"]["valid_vertex_count"],
        "valid_cells": int(valid_cells.sum()),
        "same_wrap_corridor_cells": int(labeled_cells.sum()),
        "all_review_cue_cells": int(cue_mask.sum()),
        "same_wrap_corridor_cue_cells": int(corridor_cues.sum()),
        "all_v0_1_equivalent_cue_cells": int(legacy_cues.sum()),
        "same_wrap_corridor_v0_1_equivalent_cue_cells": int(
            corridor_legacy_cues.sum()
        ),
        "all_coherent_normal_step_cells": int(normal_step_mask.sum()),
        "same_wrap_corridor_coherent_normal_step_cells": int(
            corridor_normal_steps.sum()
        ),
        "has_any_review_cue": bool(cue_mask.any()),
        "has_review_cue_in_same_wrap_corridor": bool(corridor_cues.any()),
        "has_v0_1_equivalent_cue": bool(legacy_cues.any()),
        "has_v0_1_equivalent_cue_in_same_wrap_corridor": bool(
            corridor_legacy_cues.any()
        ),
        "has_coherent_normal_step": bool(normal_step_mask.any()),
        "has_coherent_normal_step_in_same_wrap_corridor": bool(
            corridor_normal_steps.any()
        ),
        "finding_codes": finding_codes,
        "audit_signature": _audit_signature(report),
        "files_sha256": file_hashes,
    }
    return observation, data, report, labeled_cells


def _synthetic_observations(
    patch_id: str,
    data: Any,
    base_report: dict[str, Any],
    config: AuditConfig,
    offsets: tuple[float, ...],
    widths: tuple[int, ...],
) -> list[dict[str, Any]]:
    base_cues = np.asarray(base_report["_arrays"]["review_cue_mask"], dtype=bool)
    base_normal_steps = np.asarray(
        base_report["_arrays"]["coherent_normal_step_cells"],
        dtype=bool,
    )
    base_legacy_cues = base_cues & ~base_normal_steps
    observations: list[dict[str, Any]] = []

    null_case = inject_normal_offset_switch(
        data,
        offset_voxels=0.0,
        transition_width_cells=widths[0],
    )
    null_report = audit_mesh(null_case.data, config)
    observations.append(
        {
            "patch_id": patch_id,
            "kind": "null",
            "offset_voxels": 0.0,
            "transition_width_cells": widths[0],
            "orientation": null_case.orientation,
            "seam_index": null_case.seam_index,
            "seam_cells": int(null_case.seam_cells.sum()),
            "evaluation_cells": int(null_case.evaluation_cells.sum()),
            "signature_matches_baseline": (
                _audit_signature(null_report) == _audit_signature(base_report)
            ),
        }
    )

    for offset in offsets:
        for width in widths:
            case = inject_normal_offset_switch(
                data,
                offset_voxels=offset,
                transition_width_cells=width,
            )
            report = audit_mesh(case.data, config)
            cues = np.asarray(report["_arrays"]["review_cue_mask"], dtype=bool)
            normal_steps = np.asarray(
                report["_arrays"]["coherent_normal_step_cells"],
                dtype=bool,
            )
            localized = cues & case.evaluation_cells
            incremental = cues & ~base_cues
            incremental_localized = incremental & case.evaluation_cells
            incremental_normal_steps = normal_steps & ~base_normal_steps
            incremental_localized_normal_steps = (
                incremental_normal_steps & case.evaluation_cells
            )
            legacy_cues = cues & ~normal_steps
            incremental_legacy_cues = legacy_cues & ~base_legacy_cues
            incremental_localized_legacy_cues = (
                incremental_legacy_cues & case.evaluation_cells
            )
            observations.append(
                {
                    "patch_id": patch_id,
                    "kind": "normal-offset-proxy",
                    "offset_voxels": float(offset),
                    "nominal_winding_fraction": float(offset / 16.0),
                    "transition_width_cells": int(width),
                    "orientation": case.orientation,
                    "seam_index": case.seam_index,
                    "seam_cells": int(case.seam_cells.sum()),
                    "evaluation_cells": int(case.evaluation_cells.sum()),
                    "all_cue_cells": int(cues.sum()),
                    "localized_cue_cells": int(localized.sum()),
                    "incremental_cue_cells": int(incremental.sum()),
                    "incremental_localized_cue_cells": int(
                        incremental_localized.sum()
                    ),
                    "incremental_coherent_normal_step_cells": int(
                        incremental_normal_steps.sum()
                    ),
                    "incremental_localized_coherent_normal_step_cells": int(
                        incremental_localized_normal_steps.sum()
                    ),
                    "incremental_v0_1_equivalent_cue_cells": int(
                        incremental_legacy_cues.sum()
                    ),
                    "incremental_localized_v0_1_equivalent_cue_cells": int(
                        incremental_localized_legacy_cues.sum()
                    ),
                    "raw_case_detected": bool(localized.any()),
                    "incremental_case_detected": bool(incremental_localized.any()),
                    "incremental_coherent_normal_step_case_detected": bool(
                        incremental_localized_normal_steps.any()
                    ),
                    "incremental_v0_1_equivalent_case_detected": bool(
                        incremental_localized_legacy_cues.any()
                    ),
                    "finding_codes": [
                        finding["code"] for finding in report["findings"]
                    ],
                }
            )
    return observations


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _aggregate(
    base: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
) -> dict[str, Any]:
    labeled_base = [
        item
        for item in base
        if item["same_wrap_annotation_points"] > 0
        and item["same_wrap_corridor_cells"] > 0
    ]
    total_corridor = sum(
        item["same_wrap_corridor_cells"] for item in labeled_base
    )
    total_corridor_cues = sum(
        item["same_wrap_corridor_cue_cells"] for item in labeled_base
    )
    total_corridor_normal_steps = sum(
        item["same_wrap_corridor_coherent_normal_step_cells"]
        for item in labeled_base
    )
    total_corridor_legacy_cues = sum(
        item["same_wrap_corridor_v0_1_equivalent_cue_cells"]
        for item in labeled_base
    )
    patch_corridor_cues = sum(
        item["has_review_cue_in_same_wrap_corridor"] for item in labeled_base
    )
    patch_corridor_normal_steps = sum(
        item["has_coherent_normal_step_in_same_wrap_corridor"]
        for item in labeled_base
    )
    patch_corridor_legacy_cues = sum(
        item["has_v0_1_equivalent_cue_in_same_wrap_corridor"]
        for item in labeled_base
    )
    finding_prevalence = Counter(
        code for item in base for code in set(item["finding_codes"])
    )
    negative = {
        "audited_patches": len(base),
        "patches_with_labeled_corridors": len(labeled_base),
        "same_wrap_annotation_points": sum(
            item["same_wrap_annotation_points"] for item in labeled_base
        ),
        "same_wrap_corridor_cells": total_corridor,
        "same_wrap_corridor_cue_cells": total_corridor_cues,
        "corridor_cue_cell_rate": (
            total_corridor_cues / total_corridor if total_corridor else None
        ),
        "same_wrap_corridor_coherent_normal_step_cells": (
            total_corridor_normal_steps
        ),
        "corridor_coherent_normal_step_cell_rate": (
            total_corridor_normal_steps / total_corridor
            if total_corridor
            else None
        ),
        "same_wrap_corridor_v0_1_equivalent_cue_cells": (
            total_corridor_legacy_cues
        ),
        "corridor_v0_1_equivalent_cue_cell_rate": (
            total_corridor_legacy_cues / total_corridor
            if total_corridor
            else None
        ),
        "patches_with_corridor_cues": patch_corridor_cues,
        "patch_corridor_cue_rate": (
            patch_corridor_cues / len(labeled_base) if labeled_base else None
        ),
        "patch_corridor_cue_rate_wilson_95": _wilson(
            patch_corridor_cues,
            len(labeled_base),
        ),
        "patches_with_corridor_coherent_normal_steps": (
            patch_corridor_normal_steps
        ),
        "patch_corridor_coherent_normal_step_rate": (
            patch_corridor_normal_steps / len(labeled_base)
            if labeled_base
            else None
        ),
        "patch_corridor_coherent_normal_step_rate_wilson_95": _wilson(
            patch_corridor_normal_steps,
            len(labeled_base),
        ),
        "patches_with_corridor_v0_1_equivalent_cues": (
            patch_corridor_legacy_cues
        ),
        "patch_corridor_v0_1_equivalent_cue_rate": (
            patch_corridor_legacy_cues / len(labeled_base)
            if labeled_base
            else None
        ),
        "patch_corridor_v0_1_equivalent_cue_rate_wilson_95": _wilson(
            patch_corridor_legacy_cues,
            len(labeled_base),
        ),
        "patches_with_any_cue_anywhere": sum(
            item["has_any_review_cue"] for item in base
        ),
        "patches_with_coherent_normal_steps_anywhere": sum(
            item["has_coherent_normal_step"] for item in base
        ),
        "patches_with_v0_1_equivalent_cues_anywhere": sum(
            item["has_v0_1_equivalent_cue"] for item in base
        ),
        "finding_patch_prevalence": dict(sorted(finding_prevalence.items())),
    }

    nulls = [item for item in synthetic if item["kind"] == "null"]
    groups: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for item in synthetic:
        if item["kind"] == "normal-offset-proxy":
            groups[(item["offset_voxels"], item["transition_width_cells"])].append(
                item
            )
    positive_groups = []
    for (offset, width), cases in sorted(groups.items()):
        raw_detected = sum(item["raw_case_detected"] for item in cases)
        incremental_detected = sum(
            item["incremental_case_detected"] for item in cases
        )
        normal_step_detected = sum(
            item["incremental_coherent_normal_step_case_detected"]
            for item in cases
        )
        legacy_detected = sum(
            item["incremental_v0_1_equivalent_case_detected"]
            for item in cases
        )
        positive_groups.append(
            {
                "offset_voxels": offset,
                "nominal_winding_fraction": offset / 16.0,
                "transition_width_cells": width,
                "cases": len(cases),
                "raw_cases_detected": raw_detected,
                "raw_case_recall": raw_detected / len(cases),
                "raw_case_recall_wilson_95": _wilson(raw_detected, len(cases)),
                "incremental_cases_detected": incremental_detected,
                "incremental_case_recall": incremental_detected / len(cases),
                "incremental_case_recall_wilson_95": _wilson(
                    incremental_detected,
                    len(cases),
                ),
                "incremental_coherent_normal_step_cases_detected": (
                    normal_step_detected
                ),
                "incremental_coherent_normal_step_case_recall": (
                    normal_step_detected / len(cases)
                ),
                "incremental_coherent_normal_step_case_recall_wilson_95": (
                    _wilson(normal_step_detected, len(cases))
                ),
                "incremental_v0_1_equivalent_cases_detected": legacy_detected,
                "incremental_v0_1_equivalent_case_recall": (
                    legacy_detected / len(cases)
                ),
                "incremental_v0_1_equivalent_case_recall_wilson_95": (
                    _wilson(legacy_detected, len(cases))
                ),
                "evaluation_cells": sum(item["evaluation_cells"] for item in cases),
                "incremental_localized_cue_cells": sum(
                    item["incremental_localized_cue_cells"] for item in cases
                ),
                "incremental_localized_coherent_normal_step_cells": sum(
                    item["incremental_localized_coherent_normal_step_cells"]
                    for item in cases
                ),
                "incremental_localized_v0_1_equivalent_cue_cells": sum(
                    item["incremental_localized_v0_1_equivalent_cue_cells"]
                    for item in cases
                ),
            }
        )
    return {
        "reviewed_same_wrap_negative_control": negative,
        "synthetic_null_control": {
            "cases": len(nulls),
            "signature_mismatches": sum(
                not item["signature_matches_baseline"] for item in nulls
            ),
        },
        "normal_offset_proxy": positive_groups,
    }


def _select_evenly(paths: list[Path], limit: int) -> set[str]:
    if limit <= 0 or not paths:
        return set()
    if limit >= len(paths):
        return {path.name for path in paths}
    indices = np.linspace(0, len(paths) - 1, num=limit, dtype=np.int64)
    return {paths[int(index)].name for index in indices}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    offsets = _parse_float_tuple(args.offsets)
    widths = _parse_int_tuple(args.transition_widths)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")
    if args.corridor_radius < 0:
        raise SystemExit("--corridor-radius must be non-negative")
    if not args.data.is_dir():
        raise SystemExit(f"data root not found: {args.data}")

    candidates = [
        path
        for path in sorted(args.data.glob("same_wrap*"))
        if path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)
    ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("no complete same_wrap patches with correlation results found")
    synthetic_ids = _select_evenly(candidates, args.synthetic_limit)
    config = AuditConfig()

    base_observations: list[dict[str, Any]] = []
    synthetic_observations: list[dict[str, Any]] = []
    failed_paths: list[tuple[Path, Exception]] = []

    def run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        observation, data, report, _corridor = _base_observation(
            path,
            config,
            args.corridor_radius,
        )
        generated = (
            _synthetic_observations(
                path.name,
                data,
                report,
                config,
                offsets,
                widths,
            )
            if path.name in synthetic_ids
            else []
        )
        return observation, generated

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, path): path for path in candidates}
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                observation, generated = future.result()
            except Exception as exc:
                failed_paths.append((path, exc))
            else:
                base_observations.append(observation)
                synthetic_observations.extend(generated)
            if index % 50 == 0 or index == len(futures):
                print(
                    f"processed {index}/{len(futures)} "
                    f"({len(failed_paths)} pending retry)",
                    file=sys.stderr,
                )

    failures: list[dict[str, str]] = []
    for path, first_error in failed_paths:
        error: Exception = first_error
        for attempt in range(1, args.retries + 1):
            try:
                observation, generated = run(path)
            except Exception as exc:
                error = exc
                print(
                    f"retry {attempt}/{args.retries} failed for {path.name}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            else:
                base_observations.append(observation)
                synthetic_observations.extend(generated)
                print(
                    f"retry {attempt}/{args.retries} recovered {path.name}",
                    file=sys.stderr,
                )
                break
        else:
            failures.append(
                {
                    "id": path.name,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )

    base_observations.sort(key=lambda item: item["id"])
    synthetic_observations.sort(
        key=lambda item: (
            item["patch_id"],
            item["kind"],
            item["offset_voxels"],
            item["transition_width_cells"],
        )
    )
    failures.sort(key=lambda item: item["id"])

    tree_digest = hashlib.sha256()
    for item in base_observations:
        for filename, digest in sorted(item["files_sha256"].items()):
            tree_digest.update(f"{item['id']}/{filename}\0{digest}\n".encode())

    result = {
        "schema_version": "reviewed-same-wrap-benchmark-v1",
        "tool": {"name": "tifxyz-doctor", "version": "0.1.0"},
        "experiment_commits": {
            "frozen_detector_and_protocol": (
                "d3c8309ca707e2f18e7d64e38fb7be4ff4ca77c0"
            ),
            "cue_specific_reporting": (
                "b4d70220b3c4ab33dcf53f0185749b2e751a61bd"
            ),
        },
        "source": {
            "bucket": SOURCE_BUCKET,
            "selection": (
                "Complete directories named same_wrap* with meta.json, "
                "corr_points_results.json, x.tif, y.tif, and z.tif."
            ),
            "download_command": (
                "hf buckets sync "
                f"{SOURCE_BUCKET} DATA "
                "--include 'same_wrap*/meta.json' "
                "--include 'same_wrap*/corr_points_results.json' "
                "--include 'same_wrap*/x.tif' "
                "--include 'same_wrap*/y.tif' "
                "--include 'same_wrap*/z.tif'"
            ),
            "selected_patches": len(candidates),
            "successfully_audited_patches": len(base_observations),
            "tree_sha256": tree_digest.hexdigest(),
        },
        "source_data_license": {
            "spdx": "CC-BY-NC-4.0",
            "url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "notice": (
                "Downloaded Vesuvius data remains externally licensed and is "
                "not redistributed by this repository."
            ),
        },
        "configuration": {
            "doctor": asdict(config),
            "same_wrap_corridor_radius_cells": args.corridor_radius,
            "synthetic_patch_selection": (
                "Evenly spaced indices across the sorted selected patch IDs."
            ),
            "synthetic_patch_limit": args.synthetic_limit,
            "nominal_initial_dr_per_winding_voxels": 16.0,
            "normal_offsets_voxels": list(offsets),
            "transition_widths_cells": list(widths),
            "synthetic_evaluation_radius_cells": 1,
        },
        "interpretation": {
            "negative_control": (
                "The reviewed same-wrap annotation supports a negative label only "
                "inside its mapped corridor. It does not prove that every other "
                "cell in the patch is globally correct."
            ),
            "positive_control": (
                "The normal-offset cases are controlled proxies built from real "
                "reviewed surfaces. They test sensitivity to a known seam but are "
                "not a labeled sample of naturally occurring tracer failures."
            ),
            "threshold_policy": (
                "Doctor's published v0.1 defaults are used unchanged. The offset "
                "and width ladders are declared in configuration, including the "
                "zero-offset null."
            ),
            "v0_1_equivalent": (
                "The v0.1-equivalent mask is the union cue mask with only the "
                "new coherent-normal-step cells removed. The new cue does not "
                "alter any pre-existing metric, threshold, or cue family."
            ),
        },
        "aggregate": _aggregate(base_observations, synthetic_observations),
        "failures": failures,
        "patches": base_observations,
        "synthetic_cases": synthetic_observations,
    }
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"({len(base_observations)} patches, "
        f"{len(synthetic_observations)} synthetic cases, "
        f"{len(failures)} failures)"
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
