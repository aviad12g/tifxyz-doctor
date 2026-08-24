#!/usr/bin/env python3
"""Test uniform normal-offset salvage for one small TIFFXYZ patch.

Every integer shift in [-15, 15] is applied to the original valid vertices
along the original positive stored-grid normal field.  Coordinates are rounded
to the float32 TIFFXYZ storage precision, normals are recomputed from the
shifted geometry, and public m7 is actually resampled over new offsets
[-15, 15].  No old stack is rolled or reused.

A corrected copy is materialized only when one constant shift passes geometry,
sample validity, >=90% five-offset support, 100% seed-core support, deterministic
50/50 transects, and the stronger exhaustive all-valid-vertex transect check.
The original patch is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import tifffile

from native_surface_sampler import SurfaceGrid, estimate_surface_normals, validate_surface_geometry
from preflight_debug_m7 import (
    DEFAULT_SUPPORT_OFFSETS,
    DEFAULT_TRANSECT_OFFSETS,
    DEFAULT_VOLUME_SHAPE_ZYX,
    DEFAULT_VOXEL_UM,
    derive_geometry_masks,
    sample_offsets,
)
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import ZarrV2ArraySpec, fetch_json, sha256_file
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_public_m7_patch import (
    DEFAULT_M7_ROOT,
    DEFAULT_PATCH,
    PublicZarrChunkSampler,
    _array_sha256,
    _atomic_json,
    evaluate_all_vertex_transects,
    evaluate_valid_vertex_transects,
    seed_surface_alignment,
    support_summary,
    validate_metadata_axes,
)


DEFAULT_OUTPUT_DIR = Path(
    "outputs/first-letters-geometry/vc3d-interactive/"
    "priority1-seed-3526-4188-14072/growpatch-5gen-uniform-offset-sweep"
)
DEFAULT_SHIFTS = tuple(range(-15, 16))


def shifted_surface_float32(
    surface: SurfaceGrid,
    original_normals_xyz: NDArray[np.float64],
    shift_voxels: int,
) -> SurfaceGrid:
    """Apply a scalar offset along original normals at TIFFXYZ precision."""

    if original_normals_xyz.shape != (*surface.shape, 3):
        raise ValueError("normal field shape differs from surface")
    points = surface.points_xyz.copy()
    points[surface.valid] += float(shift_voxels) * original_normals_xyz[surface.valid]
    stored = points.astype(np.float32)
    return SurfaceGrid.from_tifxyz(
        stored[..., 0], stored[..., 1], stored[..., 2], mask=surface.valid
    )


def hard_geometry_summary(
    surface: SurfaceGrid,
    *,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
) -> tuple[Any, dict[str, Any]]:
    masks = derive_geometry_masks(
        surface,
        volume_shape_zyx=volume_shape_zyx,
        voxel_um=voxel_um,
        maximum_offset_voxels=15,
    )
    offsets_um = [offset * voxel_um for offset in DEFAULT_TRANSECT_OFFSETS]
    reports = {
        label: validate_surface_geometry(
            surface,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        for label, sign in (("positive", 1), ("negative", -1))
    }
    report = reports["positive"]
    counts = {
        key: int(report[key])
        for key in (
            "discontinuity_edge_count",
            "neighbor_wrap_risk_edge_count",
            "degenerate_quad_count",
            "folded_quad_count",
            "abrupt_normal_flip_edge_count",
            "distorted_quad_count",
        )
    }
    passed = (
        all(value == 0 for value in counts.values())
        and float(report["in_bounds_vertex_fraction"]) == 1.0
        and float(report["normal_valid_vertex_fraction"]) == 1.0
        and float(report["offset_sample_in_bounds_fraction"]) == 1.0
    )
    return masks, {
        "pass": passed,
        "hard_defect_counts": counts,
        "native_reports_by_orientation": reports,
        "localized_masks": masks.summary,
        "self_intersection_status": "not evaluated; unknown, never assumed zero",
    }


def generation_metrics_compact(
    generations: NDArray[Any] | None,
    valid: NDArray[np.bool_],
    support_frames: NDArray[np.uint8],
    all_frames: NDArray[np.uint8],
) -> tuple[dict[str, Any] | None, float]:
    if generations is None:
        support = support_summary(support_frames, DEFAULT_SUPPORT_OFFSETS, valid)
        return None, float(support["support_fraction"])
    if generations.shape != valid.shape:
        raise ValueError("generations.tif shape differs from shifted surface")
    values = sorted(int(value) for value in np.unique(generations[valid]))
    minimum = min(values) if values else None
    by_generation: dict[str, Any] = {}
    for value in values:
        mask = valid & (generations == value)
        support = support_summary(support_frames, DEFAULT_SUPPORT_OFFSETS, mask)
        transects = evaluate_all_vertex_transects(all_frames, mask)
        by_generation[str(value)] = {
            "vertex_count": int(np.count_nonzero(mask)),
            "support_fraction": float(support["support_fraction"]),
            "offset0_support_fraction": float(support["offset0_support_fraction"]),
            "transect_clean_fraction": float(transects["clean_fraction"]),
            "transect_category_counts": transects["category_counts"],
        }
    seed_mask = valid & (generations == minimum) if minimum is not None else np.zeros_like(valid)
    seed_support = support_summary(
        support_frames, DEFAULT_SUPPORT_OFFSETS, seed_mask
    )
    seed_transects = evaluate_all_vertex_transects(all_frames, seed_mask)
    report = {
        "valid_generation_values": values,
        "minimum_valid_generation_interpreted_as_seed_core": minimum,
        "generation_zero_valid_vertex_count": int(np.count_nonzero(valid & (generations == 0))),
        "seed_core": {
            "vertex_count": int(np.count_nonzero(seed_mask)),
            "support": seed_support,
            "transects": seed_transects,
        },
        "by_generation": by_generation,
    }
    return report, float(seed_support["support_fraction"])


def evaluate_shifted_surface(
    surface: SurfaceGrid,
    sampler: Any,
    *,
    shift_voxels: int,
    volume_shape_zyx: Sequence[int],
    voxel_um: float,
    generations: NDArray[Any] | None,
    metadata_seed_xyz: Sequence[float] | None,
    rng_seed: int,
    minimum_support: float,
    minimum_seed_support: float,
) -> tuple[dict[str, Any], NDArray[np.uint8], NDArray[np.bool_]]:
    masks, geometry = hard_geometry_summary(
        surface, volume_shape_zyx=volume_shape_zyx, voxel_um=voxel_um
    )
    all_frames, all_sample_valid = sample_offsets(
        sampler,
        surface,
        masks.normals_xyz,
        masks.sample_valid,
        DEFAULT_TRANSECT_OFFSETS,
    )
    requested = np.broadcast_to(masks.sample_valid, all_sample_valid.shape)
    sampling_complete = bool(np.all(all_sample_valid[requested]))
    support_indices = [
        DEFAULT_TRANSECT_OFFSETS.index(offset) for offset in DEFAULT_SUPPORT_OFFSETS
    ]
    support_frames = all_frames[support_indices]
    support = support_summary(
        support_frames, DEFAULT_SUPPORT_OFFSETS, masks.sample_valid
    )
    deterministic = evaluate_valid_vertex_transects(
        all_frames, masks.sample_valid, seed=rng_seed, sample_count=50
    )
    exhaustive = evaluate_all_vertex_transects(all_frames, masks.sample_valid)
    generation_report, seed_support_fraction = generation_metrics_compact(
        generations,
        masks.sample_valid,
        support_frames,
        all_frames,
    )
    seed_alignment = (
        seed_surface_alignment(surface, metadata_seed_xyz, voxel_um=voxel_um)
        if metadata_seed_xyz is not None
        else None
    )
    gate_results = [
        {"name": "hard_geometry", "pass": bool(geometry["pass"])},
        {"name": "all_requested_m7_samples_valid", "pass": sampling_complete},
        {
            "name": "five_offset_m7_support",
            "threshold": minimum_support,
            "value": float(support["support_fraction"]),
            "pass": float(support["support_fraction"]) >= minimum_support,
        },
        {
            "name": "seed_core_five_offset_m7_support",
            "threshold": minimum_seed_support,
            "value": seed_support_fraction,
            "pass": seed_support_fraction >= minimum_seed_support,
        },
        {
            "name": "deterministic_50_of_50_clean_transects",
            "value": int(deterministic.get("pass_count", 0)),
            "threshold": 50,
            "pass": bool(deterministic.get("pass", False)),
        },
        {
            "name": "exhaustive_all_valid_vertex_transects",
            "value": int(exhaustive["category_counts"]["single_run_centered"]),
            "threshold": int(exhaustive["eligible_vertex_count"]),
            "pass": bool(exhaustive["all_clean"]),
        },
    ]
    passed = all(bool(gate["pass"]) for gate in gate_results)
    valid_points = surface.points_xyz[surface.valid]
    record = {
        "shift_voxels": int(shift_voxels),
        "shift_formula": "corrected_xyz = original_xyz + shift_voxels * original_positive_normal_xyz",
        "pass_all_measured_gates": passed,
        "gate_results": gate_results,
        "coordinate_bounds_xyz": {
            "minimum": np.min(valid_points, axis=0).tolist(),
            "maximum": np.max(valid_points, axis=0).tolist(),
        },
        "surface_points_content_sha256": _array_sha256(surface.points_xyz),
        "geometry": geometry,
        "sampling_complete": sampling_complete,
        "m7_support": support,
        "deterministic_50_transects": deterministic,
        "exhaustive_transects": exhaustive,
        "generation_support": generation_report,
        "metadata_seed_surface_alignment": seed_alignment,
        "quantized_frames_content_sha256": _array_sha256(all_frames),
        "sample_validity_content_sha256": _array_sha256(all_sample_valid),
    }
    return record, all_frames, all_sample_valid


def shift_rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Deterministic descending quality key; larger tuples are better."""

    support = record["m7_support"]
    seed_support = record.get("generation_support")
    seed_fraction = (
        float(seed_support["seed_core"]["support"]["support_fraction"])
        if seed_support is not None
        else float(support["support_fraction"])
    )
    deterministic = record["deterministic_50_transects"]
    exhaustive = record["exhaustive_transects"]
    alignment = record.get("metadata_seed_surface_alignment")
    distance = (
        float(alignment["distance_voxels"])
        if isinstance(alignment, Mapping) and alignment.get("evaluated")
        else math.inf
    )
    shift = int(record["shift_voxels"])
    gates = list(record.get("gate_results", ()))
    gate_pass_count = sum(int(bool(gate.get("pass", False))) for gate in gates)
    gate_attainment: list[float] = []
    for gate in gates:
        if "value" in gate and "threshold" in gate:
            threshold = float(gate["threshold"])
            value = float(gate["value"])
            gate_attainment.append(min(1.0, value / threshold) if threshold > 0 else 1.0)
        else:
            gate_attainment.append(1.0 if gate.get("pass", False) else 0.0)
    worst_gate_attainment = min(gate_attainment) if gate_attainment else 0.0
    return (
        int(bool(record["pass_all_measured_gates"])),
        gate_pass_count,
        worst_gate_attainment,
        int(deterministic.get("pass_count", 0)),
        float(exhaustive["clean_fraction"]),
        float(support["support_fraction"]),
        seed_fraction,
        float(support["offset0_support_fraction"]),
        -distance,
        -abs(shift),
        -shift,
    )


def surface_area_cm2(surface: SurfaceGrid, voxel_um: float) -> float:
    points = surface.points_xyz * voxel_um
    total_um2 = 0.0
    for row in range(surface.shape[0] - 1):
        for column in range(surface.shape[1] - 1):
            if not bool(surface.valid[row : row + 2, column : column + 2].all()):
                continue
            p00 = points[row, column]
            p01 = points[row, column + 1]
            p10 = points[row + 1, column]
            p11 = points[row + 1, column + 1]
            total_um2 += 0.5 * float(np.linalg.norm(np.cross(p01 - p00, p11 - p00)))
            total_um2 += 0.5 * float(np.linalg.norm(np.cross(p11 - p00, p10 - p00)))
    return total_um2 / 1e8


def signed_shift_label(shift: int) -> str:
    if shift < 0:
        return f"m{abs(shift):02d}"
    if shift > 0:
        return f"p{shift:02d}"
    return "zero"


def materialize_corrected_copy(
    source_dir: Path,
    destination_parent: Path,
    surface: SurfaceGrid,
    *,
    shift_voxels: int,
    voxel_um: float,
    original_normals_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Write a new TIFFXYZ asset while proving the source remained unchanged."""

    source_files = [source_dir / name for name in ("meta.json", "x.tif", "y.tif", "z.tif")]
    generations_path = source_dir / "generations.tif"
    if generations_path.exists():
        source_files.append(generations_path)
    source_hashes_before = {path.name: sha256_file(path) for path in source_files}
    original_meta = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
    original_uuid = str(original_meta.get("uuid", source_dir.name))
    label = signed_shift_label(shift_voxels)
    destination = destination_parent / "corrected" / f"{original_uuid}_normal_offset_{label}"
    destination.mkdir(parents=True, exist_ok=False)

    points = surface.points_xyz.astype(np.float32)
    for component, index in (("x", 0), ("y", 1), ("z", 2)):
        tifffile.imwrite(
            destination / f"{component}.tif",
            points[..., index],
            photometric="minisblack",
            metadata=None,
        )
    if generations_path.exists():
        shutil.copyfile(generations_path, destination / "generations.tif")
    valid_points = surface.points_xyz[surface.valid]
    corrected_meta = dict(original_meta)
    corrected_meta.update(
        {
            "uuid": f"{original_uuid}_normal_offset_{label}",
            "source": "deterministic_uniform_normal_offset_salvage",
            "source_uuid": original_uuid,
            "area_cm2": surface_area_cm2(surface, voxel_um),
            "bbox": [
                np.min(valid_points, axis=0).tolist(),
                np.max(valid_points, axis=0).tolist(),
            ],
            "uniform_normal_offset_salvage": {
                "shift_voxels": int(shift_voxels),
                "shift_um": float(shift_voxels * voxel_um),
                "normal_orientation": "original positive stored-grid normal",
                "formula": "corrected_xyz = original_xyz + shift_voxels * original_positive_normal_xyz",
                "coordinate_storage_dtype": "float32",
                "original_normals_content_sha256": original_normals_sha256,
                "source_file_sha256": source_hashes_before,
            },
        }
    )
    (destination / "meta.json").write_text(
        json.dumps(corrected_meta, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_hashes_after = {path.name: sha256_file(path) for path in source_files}
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("source patch changed while materializing corrected copy")
    output_files = sorted(path for path in destination.iterdir() if path.is_file())
    output_hashes = {path.name: sha256_file(path) for path in output_files}
    reloaded = load_tifxyz_asset(destination, resolution="stored")
    np.testing.assert_array_equal(
        reloaded.surface.points_xyz.astype(np.float32), points
    )
    np.testing.assert_array_equal(reloaded.surface.valid, surface.valid)
    provenance = {
        "source_directory": str(source_dir.resolve()),
        "source_file_sha256_before": source_hashes_before,
        "source_file_sha256_after": source_hashes_after,
        "source_unchanged": True,
        "destination_directory": str(destination.resolve()),
        "output_file_sha256": output_hashes,
        "roundtrip_coordinate_match": True,
    }
    _atomic_json(destination / "materialization-provenance.json", provenance)
    provenance["output_file_sha256"]["materialization-provenance.json"] = sha256_file(
        destination / "materialization-provenance.json"
    )
    return destination, provenance


def compact_shift_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    generation = record.get("generation_support")
    alignment = record.get("metadata_seed_surface_alignment")
    gates = list(record.get("gate_results", ()))
    gate_pass_count = sum(int(bool(gate.get("pass", False))) for gate in gates)
    attainments: list[float] = []
    for gate in gates:
        if "value" in gate and "threshold" in gate:
            threshold = float(gate["threshold"])
            attainments.append(
                min(1.0, float(gate["value"]) / threshold) if threshold > 0 else 1.0
            )
        else:
            attainments.append(1.0 if gate.get("pass", False) else 0.0)
    return {
        "shift_voxels": int(record["shift_voxels"]),
        "pass_all_measured_gates": bool(record["pass_all_measured_gates"]),
        "passed_gate_count": gate_pass_count,
        "measured_gate_count": len(gates),
        "worst_gate_attainment_fraction": min(attainments) if attainments else 0.0,
        "hard_geometry_pass": bool(record["geometry"]["pass"]),
        "sampling_complete": bool(record["sampling_complete"]),
        "support_fraction": float(record["m7_support"]["support_fraction"]),
        "offset0_support_fraction": float(record["m7_support"]["offset0_support_fraction"]),
        "seed_core_support_fraction": (
            float(generation["seed_core"]["support"]["support_fraction"])
            if generation is not None
            else None
        ),
        "deterministic_transect_pass_count": int(
            record["deterministic_50_transects"].get("pass_count", 0)
        ),
        "exhaustive_clean_count": int(
            record["exhaustive_transects"]["category_counts"]["single_run_centered"]
        ),
        "exhaustive_eligible_count": int(
            record["exhaustive_transects"]["eligible_vertex_count"]
        ),
        "seed_to_surface_distance_voxels": (
            float(alignment["distance_voxels"])
            if isinstance(alignment, Mapping) and alignment.get("evaluated")
            else None
        ),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = Path(args.tifxyz)
    asset = load_tifxyz_asset(source_dir, resolution="stored")
    original_normals = estimate_surface_normals(
        asset.surface, voxel_size_zyx_um=(float(args.voxel_um),) * 3
    )
    if not np.array_equal(original_normals.valid, asset.surface.valid):
        raise ValueError("not every original valid vertex has an original normal")
    original_normals_hash = _array_sha256(original_normals.positive_xyz)

    generations_path = source_dir / "generations.tif"
    generations = (
        np.asarray(tifffile.imread(generations_path))
        if generations_path.exists()
        else None
    )
    metadata_seed_raw = asset.metadata.get("seed")
    metadata_seed = (
        tuple(float(value) for value in metadata_seed_raw)
        if isinstance(metadata_seed_raw, (list, tuple)) and len(metadata_seed_raw) == 3
        else None
    )

    root = str(args.m7_root).rstrip("/")
    array_path = str(args.m7_array_path).strip("/")
    group, group_provenance = fetch_json(root + "/.zgroup")
    array_metadata, array_provenance = fetch_json(root + f"/{array_path}/.zarray")
    attributes, attributes_provenance = fetch_json(root + "/.zattrs")
    if group.get("zarr_format") != 2:
        raise ValueError("m7 root is not Zarr v2")
    validate_metadata_axes(attributes, array_path)
    spec = ZarrV2ArraySpec.from_metadata(array_metadata)
    if spec.shape_zyx != tuple(int(value) for value in args.volume_shape_zyx):
        raise ValueError("m7 shape differs from expected PHerc1447 shape")
    if spec.chunks_zyx != (192, 192, 192):
        raise ValueError("m7 chunk shape differs from expected 192^3")
    if spec.compressor is None or spec.compressor.get("id") != "blosc":
        raise ValueError("m7 is not Blosc-compressed")
    sampler = PublicZarrChunkSampler(
        root_url=root,
        array_path=array_path,
        spec=spec,
        blosc_path=Path(args.blosc),
    )

    records: list[dict[str, Any]] = []
    artifacts: dict[int, tuple[SurfaceGrid, NDArray[np.uint8], NDArray[np.bool_]]] = {}
    for shift in DEFAULT_SHIFTS:
        shifted = shifted_surface_float32(
            asset.surface, original_normals.positive_xyz, shift
        )
        record, frames, validity = evaluate_shifted_surface(
            shifted,
            sampler,
            shift_voxels=shift,
            volume_shape_zyx=spec.shape_zyx,
            voxel_um=float(args.voxel_um),
            generations=generations,
            metadata_seed_xyz=metadata_seed,
            rng_seed=int(args.seed),
            minimum_support=float(args.minimum_support),
            minimum_seed_support=float(args.minimum_seed_support),
        )
        records.append(record)
        artifacts[shift] = (shifted, frames, validity)

    best = max(records, key=shift_rank_key)
    passing = [record for record in records if record["pass_all_measured_gates"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_shift = int(best["shift_voxels"])
    best_surface, best_frames, best_validity = artifacts[best_shift]
    best_frames_path = output_dir / "best-shift-m7-quantized-offsets-minus15-plus15.npy"
    best_validity_path = output_dir / "best-shift-m7-sample-validity.npy"
    np.save(best_frames_path, best_frames, allow_pickle=False)
    np.save(best_validity_path, best_validity, allow_pickle=False)

    materialization: dict[str, Any] | None = None
    post_materialization: dict[str, Any] | None = None
    if passing:
        destination, materialization = materialize_corrected_copy(
            source_dir,
            output_dir,
            best_surface,
            shift_voxels=best_shift,
            voxel_um=float(args.voxel_um),
            original_normals_sha256=original_normals_hash,
        )
        reloaded = load_tifxyz_asset(destination, resolution="stored")
        post_record, post_frames, post_validity = evaluate_shifted_surface(
            reloaded.surface,
            sampler,
            shift_voxels=0,
            volume_shape_zyx=spec.shape_zyx,
            voxel_um=float(args.voxel_um),
            generations=generations,
            metadata_seed_xyz=metadata_seed,
            rng_seed=int(args.seed),
            minimum_support=float(args.minimum_support),
            minimum_seed_support=float(args.minimum_seed_support),
        )
        post_materialization = {
            "pass_all_measured_gates": post_record["pass_all_measured_gates"],
            "support_fraction": post_record["m7_support"]["support_fraction"],
            "offset0_support_fraction": post_record["m7_support"]["offset0_support_fraction"],
            "deterministic_pass_count": post_record["deterministic_50_transects"].get("pass_count", 0),
            "exhaustive_clean_count": post_record["exhaustive_transects"]["category_counts"]["single_run_centered"],
            "frames_identical_to_sweep": bool(np.array_equal(post_frames, best_frames)),
            "validity_identical_to_sweep": bool(np.array_equal(post_validity, best_validity)),
        }
        if not (
            post_materialization["pass_all_measured_gates"]
            and post_materialization["frames_identical_to_sweep"]
            and post_materialization["validity_identical_to_sweep"]
        ):
            raise RuntimeError("materialized corrected copy did not reproduce passing sweep result")

    result = {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": str(args.candidate_id),
        "verdict": "uniform_offset_salvage_pass" if passing else "uniform_offset_salvage_failed",
        "go": bool(passing),
        "next_action": (
            "use only the separately materialized corrected copy for another bounded geometry step"
            if passing
            else "stop this patch; do not use per-vertex warping in this campaign step"
        ),
        "paid_compute_cost_usd": 0.0,
        "sweep": {
            "integer_shifts_tested_inclusive": [-15, 15],
            "tested_shift_count": len(records),
            "application_normal": "original positive stored-grid normal field",
            "recomputed_after_shift": [
                "float32 TIFFXYZ coordinates",
                "surface normals",
                "geometry validation",
                "public m7 offsets -15..15",
                "five-offset support",
                "seed-core support",
                "deterministic and exhaustive transects",
                "seed-to-surface distance",
            ],
            "old_frames_rolled_or_reused": False,
            "passing_shifts": [int(record["shift_voxels"]) for record in passing],
            "tie_break_order": [
                "passes all measured gates",
                "higher count of measured gates passed",
                "higher worst normalized gate attainment (numeric gates capped at 1)",
                "higher deterministic transect pass count",
                "higher exhaustive clean fraction",
                "higher five-offset support fraction",
                "higher seed-core support fraction",
                "higher offset0 support fraction",
                "smaller seed-to-surface distance",
                "smaller absolute shift",
                "smaller signed shift",
            ],
            "best_shift_voxels": best_shift,
            "best_shift_signed_orientation": (
                f"{abs(best_shift)} voxels along "
                + ("positive" if best_shift > 0 else "negative" if best_shift < 0 else "zero")
                + " original normal"
            ),
            "best_shift_summary": compact_shift_summary(best),
            "compact_results": [compact_shift_summary(record) for record in records],
            "full_results": records,
        },
        "inputs": {
            "tifxyz": asset.manifest(),
            "generations": (
                {
                    "path": str(generations_path.resolve()),
                    "sha256": sha256_file(generations_path),
                    "array_content_sha256": _array_sha256(generations),
                }
                if generations is not None
                else None
            ),
            "original_positive_normals_content_sha256": original_normals_hash,
            "m7": {
                "root": root,
                "array_path": array_path,
                "group_metadata": group_provenance,
                "array_metadata": array_provenance,
                "root_attributes": attributes_provenance,
                "array_spec": {
                    "shape_zyx": list(spec.shape_zyx),
                    "chunks_zyx": list(spec.chunks_zyx),
                    "dtype": spec.dtype.str,
                    "compressor": spec.compressor,
                },
                "codec": {
                    "path": str(Path(args.blosc).resolve()),
                    "sha256": sha256_file(Path(args.blosc)),
                },
            },
        },
        "public_m7_sampling": sampler.manifest(),
        "best_shift_arrays": {
            "quantized_frames": {
                "path": str(best_frames_path.resolve()),
                "sha256": sha256_file(best_frames_path),
                "array_content_sha256": _array_sha256(best_frames),
            },
            "sample_validity": {
                "path": str(best_validity_path.resolve()),
                "sha256": sha256_file(best_validity_path),
                "array_content_sha256": _array_sha256(best_validity),
            },
        },
        "materialization": materialization,
        "post_materialization_validation": post_materialization,
        "limitations": [
            "Only one global scalar normal offset was tested; no per-vertex warping was attempted.",
            "Stored-grid checks do not evaluate self-intersection.",
            "A passing correction permits only another bounded geometry step, not ink inference.",
            "The tiny patch remains far below the final 0.5 cm2 area minimum.",
        ],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-id", default="priority1-seed-3525-4191-14071-growpatch-5gen-uniform-offset"
    )
    parser.add_argument("--tifxyz", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--m7-root", default=DEFAULT_M7_ROOT)
    parser.add_argument("--m7-array-path", default="0")
    parser.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="uniform-offset-sweep.json")
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument(
        "--volume-shape-zyx",
        type=int,
        nargs=3,
        default=DEFAULT_VOLUME_SHAPE_ZYX,
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--minimum-support", type=float, default=0.90)
    parser.add_argument("--minimum-seed-support", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1447)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.output_dir) / str(args.output_name)
    try:
        if not 0 <= args.minimum_support <= 1:
            raise ValueError("--minimum-support must be in [0,1]")
        if not 0 <= args.minimum_seed_support <= 1:
            raise ValueError("--minimum-seed-support must be in [0,1]")
        result = run_sweep(args)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "status": "error",
            "candidate_id": str(args.candidate_id),
            "verdict": "uniform_offset_salvage_failed",
            "go": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "paid_compute_cost_usd": 0.0,
        }
        _atomic_json(output_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1
    _atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "go": result["go"],
                "best_shift_voxels": result["sweep"]["best_shift_voxels"],
                "passing_shifts": result["sweep"]["passing_shifts"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0 if result["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
