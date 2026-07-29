#!/usr/bin/env python3
"""Benchmark Doctor in reviewed same-wrap neighborhoods and controlled proxies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tifxyz_doctor import __version__  # noqa: E402
from tifxyz_doctor.audit import AuditConfig, audit_mesh  # noqa: E402
from tifxyz_doctor.audit import public_report  # noqa: E402
from tifxyz_doctor.io import load_tifxyz  # noqa: E402
from tifxyz_doctor.reviewed_benchmark import (  # noqa: E402
    inject_normal_offset_switch,
    same_wrap_annotation_neighborhoods,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "reviewed-same-wrap-results-v0.2.0.json"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "benchmarks" / "reviewed-same-wrap-split-v1.json"
)
DEFAULT_OFFSETS_VOXELS = (4.0, 8.0, 16.0)
DEFAULT_TRANSITION_WIDTHS = (1, 4, 12)
DEFAULT_ANNOTATION_RADIUS = 4
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
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Sequential retries for transient file-read failures",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--annotation-radius",
        type=int,
        default=DEFAULT_ANNOTATION_RADIUS,
        help="Chebyshev dilation radius around mapped annotation samples",
    )
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


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


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
    annotation_radius: int,
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
    annotation_neighborhoods = same_wrap_annotation_neighborhoods(
        corr_results,
        data.shape,
        radius=annotation_radius,
    )
    labeled_cells = annotation_neighborhoods & valid_cells
    neighborhood_cues = cue_mask & labeled_cells
    neighborhood_normal_steps = normal_step_mask & labeled_cells
    legacy_cues = np.asarray(
        report["_arrays"]["v0_1_review_cue_mask"],
        dtype=bool,
    )
    neighborhood_legacy_cues = legacy_cues & labeled_cells
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
        "same_wrap_annotation_neighborhood_cells": int(labeled_cells.sum()),
        "all_review_cue_cells": int(cue_mask.sum()),
        "same_wrap_annotation_neighborhood_cue_cells": int(
            neighborhood_cues.sum()
        ),
        "all_v0_1_cue_cells": int(legacy_cues.sum()),
        "same_wrap_annotation_neighborhood_v0_1_cue_cells": int(
            neighborhood_legacy_cues.sum()
        ),
        "all_coherent_normal_step_cells": int(normal_step_mask.sum()),
        "same_wrap_annotation_neighborhood_coherent_normal_step_cells": int(
            neighborhood_normal_steps.sum()
        ),
        "has_any_review_cue": bool(cue_mask.any()),
        "has_review_cue_in_same_wrap_annotation_neighborhood": bool(
            neighborhood_cues.any()
        ),
        "has_v0_1_cue": bool(legacy_cues.any()),
        "has_v0_1_cue_in_same_wrap_annotation_neighborhood": bool(
            neighborhood_legacy_cues.any()
        ),
        "has_coherent_normal_step": bool(normal_step_mask.any()),
        "has_coherent_normal_step_in_same_wrap_annotation_neighborhood": bool(
            neighborhood_normal_steps.any()
        ),
        "finding_codes": finding_codes,
        "audit_signature": _audit_signature(report),
        "files_sha256": file_hashes,
    }
    return observation, data, report, labeled_cells


def _synthetic_observations(
    patch_id: str,
    evaluation_split: str,
    overlap_component_id: str,
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
    base_legacy_cues = np.asarray(
        base_report["_arrays"]["v0_1_review_cue_mask"],
        dtype=bool,
    )
    observations: list[dict[str, Any]] = []

    null_case = inject_normal_offset_switch(
        data,
        offset_voxels=0.0,
        transition_width_cells=widths[0],
    )
    null_report = audit_mesh(null_case.data, config)
    baseline_public = public_report(base_report)
    null_public = public_report(null_report)
    baseline_coordinate_bytes = data.coordinates.tobytes(order="C")
    null_coordinate_bytes = null_case.data.coordinates.tobytes(order="C")
    observations.append(
        {
            "patch_id": patch_id,
            "evaluation_split": evaluation_split,
            "overlap_component_id": overlap_component_id,
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
            "coordinates_byte_identical": (
                data.coordinates.dtype == null_case.data.coordinates.dtype
                and data.coordinates.shape == null_case.data.coordinates.shape
                and baseline_coordinate_bytes == null_coordinate_bytes
            ),
            "validity_byte_identical": (
                data.valid.dtype == null_case.data.valid.dtype
                and data.valid.shape == null_case.data.valid.shape
                and data.valid.tobytes(order="C")
                == null_case.data.valid.tobytes(order="C")
            ),
            "public_report_matches_baseline": null_public == baseline_public,
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
            legacy_cues = np.asarray(
                report["_arrays"]["v0_1_review_cue_mask"],
                dtype=bool,
            )
            incremental_legacy_cues = legacy_cues & ~base_legacy_cues
            incremental_localized_legacy_cues = (
                incremental_legacy_cues & case.evaluation_cells
            )
            observations.append(
                {
                    "patch_id": patch_id,
                    "evaluation_split": evaluation_split,
                    "overlap_component_id": overlap_component_id,
                    "kind": "normal-offset-proxy",
                    "offset_voxels": float(offset),
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
                    "incremental_v0_1_cue_cells": int(
                        incremental_legacy_cues.sum()
                    ),
                    "incremental_localized_v0_1_cue_cells": int(
                        incremental_localized_legacy_cues.sum()
                    ),
                    "raw_case_detected": bool(localized.any()),
                    "incremental_case_detected": bool(incremental_localized.any()),
                    "incremental_coherent_normal_step_case_detected": bool(
                        incremental_localized_normal_steps.any()
                    ),
                    "incremental_v0_1_case_detected": bool(
                        incremental_localized_legacy_cues.any()
                    ),
                    "finding_codes": [
                        finding["code"] for finding in report["findings"]
                    ],
                }
            )
    return observations


def _component_bootstrap_rate(
    observations: list[dict[str, Any]],
    success_key: str,
    *,
    iterations: int = 10_000,
) -> list[float] | None:
    """Deterministic cluster bootstrap over overlap-connected components."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[item["overlap_component_id"]].append(item)
    components = [grouped[key] for key in sorted(grouped)]
    if not components:
        return None
    seed_material = (
        f"tifxyz-doctor-component-bootstrap-v1\0{success_key}\0"
        + "\0".join(sorted(grouped))
    )
    seed = int.from_bytes(
        hashlib.sha256(seed_material.encode()).digest()[:8],
        byteorder="big",
    )
    rng = np.random.default_rng(seed)
    rates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(components), size=len(components))
        numerator = 0
        denominator = 0
        for component_index in sampled:
            component = components[int(component_index)]
            numerator += sum(bool(item[success_key]) for item in component)
            denominator += len(component)
        rates[iteration] = numerator / denominator
    return [
        float(np.quantile(rates, 0.025)),
        float(np.quantile(rates, 0.975)),
    ]


def _same_wrap_alert_summary(
    observations: list[dict[str, Any]],
    evaluation_group: str,
) -> dict[str, Any]:
    labeled = [
        item
        for item in observations
        if item["same_wrap_annotation_points"] > 0
        and item["same_wrap_annotation_neighborhood_cells"] > 0
    ]
    total_cells = sum(
        item["same_wrap_annotation_neighborhood_cells"] for item in labeled
    )
    cue_cells = sum(
        item["same_wrap_annotation_neighborhood_cue_cells"] for item in labeled
    )
    step_cells = sum(
        item["same_wrap_annotation_neighborhood_coherent_normal_step_cells"]
        for item in labeled
    )
    v0_1_cells = sum(
        item["same_wrap_annotation_neighborhood_v0_1_cue_cells"]
        for item in labeled
    )
    cue_patches = sum(
        item["has_review_cue_in_same_wrap_annotation_neighborhood"]
        for item in labeled
    )
    step_patches = sum(
        item["has_coherent_normal_step_in_same_wrap_annotation_neighborhood"]
        for item in labeled
    )
    v0_1_patches = sum(
        item["has_v0_1_cue_in_same_wrap_annotation_neighborhood"]
        for item in labeled
    )
    finding_prevalence = Counter(
        code for item in observations for code in set(item["finding_codes"])
    )
    return {
        "evaluation_group": evaluation_group,
        "audited_patches": len(observations),
        "overlap_components": len(
            {item["overlap_component_id"] for item in observations}
        ),
        "patches_with_annotation_neighborhoods": len(labeled),
        "same_wrap_annotation_points": sum(
            item["same_wrap_annotation_points"] for item in labeled
        ),
        "annotation_neighborhood_cells": total_cells,
        "annotation_neighborhood_cue_cells": cue_cells,
        "annotation_neighborhood_cue_cell_rate": (
            cue_cells / total_cells if total_cells else None
        ),
        "annotation_neighborhood_coherent_normal_step_cells": step_cells,
        "annotation_neighborhood_coherent_normal_step_cell_rate": (
            step_cells / total_cells if total_cells else None
        ),
        "annotation_neighborhood_v0_1_cue_cells": v0_1_cells,
        "annotation_neighborhood_v0_1_cue_cell_rate": (
            v0_1_cells / total_cells if total_cells else None
        ),
        "patches_with_annotation_neighborhood_cues": cue_patches,
        "patch_annotation_neighborhood_cue_rate": (
            cue_patches / len(labeled) if labeled else None
        ),
        "patch_annotation_neighborhood_cue_rate_component_bootstrap_95": (
            _component_bootstrap_rate(
                labeled,
                "has_review_cue_in_same_wrap_annotation_neighborhood",
            )
        ),
        "patches_with_annotation_neighborhood_coherent_normal_steps": (
            step_patches
        ),
        "patch_annotation_neighborhood_coherent_normal_step_rate": (
            step_patches / len(labeled) if labeled else None
        ),
        (
            "patch_annotation_neighborhood_coherent_normal_step_rate_"
            "component_bootstrap_95"
        ): _component_bootstrap_rate(
            labeled,
            "has_coherent_normal_step_in_same_wrap_annotation_neighborhood",
        ),
        "patches_with_annotation_neighborhood_v0_1_cues": v0_1_patches,
        "patch_annotation_neighborhood_v0_1_cue_rate": (
            v0_1_patches / len(labeled) if labeled else None
        ),
        "patch_annotation_neighborhood_v0_1_cue_rate_component_bootstrap_95": (
            _component_bootstrap_rate(
                labeled,
                "has_v0_1_cue_in_same_wrap_annotation_neighborhood",
            )
        ),
        "patches_with_any_cue_anywhere": sum(
            item["has_any_review_cue"] for item in observations
        ),
        "patches_with_coherent_normal_steps_anywhere": sum(
            item["has_coherent_normal_step"] for item in observations
        ),
        "patches_with_v0_1_cues_anywhere": sum(
            item["has_v0_1_cue"] for item in observations
        ),
        "finding_patch_prevalence": dict(sorted(finding_prevalence.items())),
    }


def _aggregate(
    base: list[dict[str, Any]],
    synthetic: list[dict[str, Any]],
) -> dict[str, Any]:
    same_wrap_groups = [
        _same_wrap_alert_summary(base, "all_reviewed_patches"),
        _same_wrap_alert_summary(
            [
                item
                for item in base
                if item["evaluation_group"] == "overlap_isolated_holdout"
            ],
            "overlap_isolated_holdout",
        ),
    ]

    nulls = [item for item in synthetic if item["kind"] == "null"]
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for item in synthetic:
        if item["kind"] == "normal-offset-proxy":
            groups[
                (
                    item["evaluation_split"],
                    item["offset_voxels"],
                    item["transition_width_cells"],
                )
            ].append(item)
    positive_groups = []
    for (evaluation_split, offset, width), cases in sorted(groups.items()):
        raw_detected = sum(item["raw_case_detected"] for item in cases)
        incremental_detected = sum(
            item["incremental_case_detected"] for item in cases
        )
        normal_step_detected = sum(
            item["incremental_coherent_normal_step_case_detected"]
            for item in cases
        )
        v0_1_detected = sum(
            item["incremental_v0_1_case_detected"] for item in cases
        )
        positive_groups.append(
            {
                "evaluation_split": evaluation_split,
                "offset_voxels": offset,
                "transition_width_cells": width,
                "cases": len(cases),
                "overlap_components": len(
                    {item["overlap_component_id"] for item in cases}
                ),
                "raw_cases_detected": raw_detected,
                "raw_case_detection_rate": raw_detected / len(cases),
                "incremental_cases_detected": incremental_detected,
                "incremental_case_detection_rate": (
                    incremental_detected / len(cases)
                ),
                "incremental_case_detection_rate_component_bootstrap_95": (
                    _component_bootstrap_rate(
                        cases,
                        "incremental_case_detected",
                    )
                ),
                "incremental_coherent_normal_step_cases_detected": (
                    normal_step_detected
                ),
                "incremental_coherent_normal_step_case_detection_rate": (
                    normal_step_detected / len(cases)
                ),
                (
                    "incremental_coherent_normal_step_case_detection_rate_"
                    "component_bootstrap_95"
                ): _component_bootstrap_rate(
                    cases,
                    "incremental_coherent_normal_step_case_detected",
                ),
                "incremental_v0_1_cases_detected": v0_1_detected,
                "incremental_v0_1_case_detection_rate": (
                    v0_1_detected / len(cases)
                ),
                "incremental_v0_1_case_detection_rate_component_bootstrap_95": (
                    _component_bootstrap_rate(
                        cases,
                        "incremental_v0_1_case_detected",
                    )
                ),
                "evaluation_cells": sum(item["evaluation_cells"] for item in cases),
                "incremental_localized_cue_cells": sum(
                    item["incremental_localized_cue_cells"] for item in cases
                ),
                "incremental_localized_coherent_normal_step_cells": sum(
                    item["incremental_localized_coherent_normal_step_cells"]
                    for item in cases
                ),
                "incremental_localized_v0_1_cue_cells": sum(
                    item["incremental_localized_v0_1_cue_cells"]
                    for item in cases
                ),
            }
        )
    return {
        "reviewed_same_wrap_annotation_neighborhood_alerts": same_wrap_groups,
        "synthetic_null_control": [
            {
                "evaluation_split": evaluation_split,
                "cases": len(cases),
                "signature_mismatches": sum(
                    not item["signature_matches_baseline"] for item in cases
                ),
                "coordinate_byte_mismatches": sum(
                    not item["coordinates_byte_identical"] for item in cases
                ),
                "validity_byte_mismatches": sum(
                    not item["validity_byte_identical"] for item in cases
                ),
                "public_report_mismatches": sum(
                    not item["public_report_matches_baseline"] for item in cases
                ),
            }
            for evaluation_split in sorted(
                {item["evaluation_split"] for item in nulls}
            )
            for cases in [
                [
                    item
                    for item in nulls
                    if item["evaluation_split"] == evaluation_split
                ]
            ]
        ],
        "normal_offset_proxy": positive_groups,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    offsets = _parse_float_tuple(args.offsets)
    widths = _parse_int_tuple(args.transition_widths)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries must be non-negative")
    if args.annotation_radius < 0:
        raise SystemExit("--annotation-radius must be non-negative")
    if not args.data.is_dir():
        raise SystemExit(f"data root not found: {args.data}")
    if not args.split_manifest.is_file():
        raise SystemExit(
            f"split manifest not found: {args.split_manifest}; run "
            "scripts/build_reviewed_patch_split.py first"
        )

    full_candidates = [
        path
        for path in sorted(args.data.glob("same_wrap*"))
        if path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)
    ]
    with args.split_manifest.open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    if split_manifest.get("schema_version") != "reviewed-same-wrap-split-v1":
        raise SystemExit("unsupported reviewed-patch split manifest")
    candidate_ids = [path.name for path in full_candidates]
    expected_ids_digest = split_manifest["source"]["selected_patch_ids_sha256"]
    if _ids_sha256(candidate_ids) != expected_ids_digest:
        raise SystemExit("selected patch IDs do not match the frozen split manifest")
    config = AuditConfig()
    if split_manifest["frozen_detector"]["configuration"] != asdict(config):
        raise SystemExit("Doctor configuration differs from the frozen split manifest")
    protocol = split_manifest["protocol"]
    if list(offsets) != protocol["normal_offsets_voxels"]:
        raise SystemExit("offset ladder differs from the frozen split manifest")
    if list(widths) != protocol["transition_widths_cells"]:
        raise SystemExit("transition widths differ from the frozen split manifest")
    if (
        args.annotation_radius
        != protocol["annotation_neighborhood_radius_cells"]
    ):
        raise SystemExit("annotation radius differs from the frozen split manifest")

    split = split_manifest["split"]
    development_ids = set(split["development_ids"])
    clean_holdout_ids = set(split["clean_holdout_pool_ids"])
    selected_holdout_ids = set(split["selected_holdout_ids"])
    component_ids = split["component_ids"]
    if development_ids & selected_holdout_ids:
        raise SystemExit("split manifest reuses a development patch in holdout")
    if not (
        development_ids | set(split["development_related_excluded_ids"])
    ).isdisjoint(clean_holdout_ids):
        raise SystemExit("split manifest leaks a development component into holdout")
    synthetic_splits = {
        **{patch_id: "development" for patch_id in development_ids},
        **{patch_id: "holdout" for patch_id in selected_holdout_ids},
    }
    evaluation_groups = {
        patch_id: (
            "overlap_isolated_holdout"
            if patch_id in clean_holdout_ids
            else "development_connected"
        )
        for patch_id in candidate_ids
    }
    candidates = full_candidates
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("no complete same_wrap patches with correlation results found")

    base_observations: list[dict[str, Any]] = []
    synthetic_observations: list[dict[str, Any]] = []
    failed_paths: list[tuple[Path, Exception]] = []

    def run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        observation, data, report, _annotation_neighborhoods = _base_observation(
            path,
            config,
            args.annotation_radius,
        )
        observation["evaluation_group"] = evaluation_groups[path.name]
        observation["overlap_component_id"] = component_ids[path.name]
        generated = (
            _synthetic_observations(
                path.name,
                synthetic_splits[path.name],
                component_ids[path.name],
                data,
                report,
                config,
                offsets,
                widths,
            )
            if path.name in synthetic_splits
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
            item["evaluation_split"],
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
        "tool": {"name": "tifxyz-doctor", "version": __version__},
        "experiment_commits": {
            "frozen_detector_and_development_protocol": (
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
            "split_manifest": {
                "path": _portable_path(args.split_manifest),
                "sha256": _sha256(args.split_manifest),
                "selected_patch_ids_sha256": expected_ids_digest,
                "overlap_graph": split_manifest["source"]["overlap_graph"],
            },
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
            "same_wrap_annotation_neighborhood_radius_cells": (
                args.annotation_radius
            ),
            "synthetic_patch_selection": {
                "development": (
                    "The exact original 64-patch cohort used while developing "
                    "the cue."
                ),
                "holdout": (
                    "Whole overlap-connected components from the 492-patch clean "
                    "pool, selected by the fixed hash protocol in the committed "
                    "split manifest after the detector was frozen."
                ),
            },
            "synthetic_patch_counts": {
                "development": len(development_ids),
                "holdout": len(selected_holdout_ids),
            },
            "normal_offsets_voxels": list(offsets),
            "transition_widths_cells": list(widths),
            "synthetic_evaluation_radius_cells": 1,
            "component_bootstrap_iterations": 10_000,
        },
        "interpretation": {
            "negative_control": (
                "A reviewed same-wrap annotation negates a sheet switch only in "
                "the mapped sample neighborhoods. Other cue families can be "
                "legitimate there, so these are descriptive alert rates—not "
                "general false-positive rates."
            ),
            "positive_control": (
                "The normal-offset cases are controlled proxies built from real "
                "reviewed surfaces. They test sensitivity to an injected seam but "
                "are not naturally occurring sheet switches or a measurement of "
                "real-world sheet-switch recall."
            ),
            "threshold_policy": (
                "The coherent-normal-step threshold and original 64-patch "
                "development protocol were frozen in commit d3c8309 before the "
                "overlap-component-isolated holdout was selected. Development "
                "and holdout results are never pooled."
            ),
            "v0_1_comparator": (
                "The v0.1 mask is captured immediately before unioning the new "
                "coherent-normal-step cells, so overlap between old and new cue "
                "families is preserved correctly."
            ),
            "incremental_detection": (
                "An incremental detection is a cue inside the evaluation band "
                "that was absent in the unmodified baseline. It is a controlled "
                "benchmark event rate, not conventional recall."
            ),
            "offset_ladder": (
                "Offsets are reported only in voxels. No fixed winding fraction "
                "is inferred from them."
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
