#!/usr/bin/env python3
"""Canonical full-raster, float32-reload geometry gate for a stored TIFFXYZ mask."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile

from native_surface_sampler import SurfaceGrid, validate_surface_geometry
from render_pherc0800_rank2_raw_stack import (
    active_quads,
    physical_area_cm2,
    quad_union_vertex_mask,
    scalable_topology_report,
)
from sample_raw_ct_seed_cube import sha256_file
from tifxyz_render_pipeline import load_tifxyz_asset
from validate_pherc1203_auto_grown_candidate import (
    compact_full_geometry,
    derived_full_shape,
    full_geometry_pass,
)
from validate_public_m7_patch import _array_sha256, _atomic_json


def materialize_full(
    output: Path,
    surface: SurfaceGrid,
    *,
    candidate_id: str,
    voxel_um: float,
    parent: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    arrays = {
        "x.tif": np.asarray(surface.x, dtype=np.float32),
        "y.tif": np.asarray(surface.y, dtype=np.float32),
        "z.tif": np.asarray(surface.z, dtype=np.float32),
        "mask.tif": np.asarray(surface.valid, dtype=np.uint8) * 255,
    }
    for filename, array in arrays.items():
        tifffile.imwrite(output / filename, array, compression="deflate")
    points = surface.points_xyz[surface.valid]
    metadata = {
        "format": "tifxyz",
        "type": "seg",
        "uuid": candidate_id,
        "source": "canonical_mask_aware_linear_resample_then_float32_materialization",
        "parent_stored_tifxyz": str(parent.resolve()),
        "scale": [1.0, 1.0],
        "stored_shape": list(surface.shape),
        "area_cm2": physical_area_cm2(surface, voxel_um=voxel_um),
        "bbox": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "meta.json", metadata)
    return {
        "directory": str(output.resolve()),
        "files": {
            name: {"path": str((output / name).resolve()), "sha256": sha256_file(output / name)}
            for name in (*arrays, "meta.json")
        },
    }


def geometry_reports(
    surface: SurfaceGrid,
    *,
    voxel_um: float,
    volume_shape_zyx: Sequence[int],
) -> dict[str, Any]:
    offsets = tuple(range(-46, 47))
    offsets_um = tuple(float(value) * voxel_um for value in offsets)
    reports: dict[str, Any] = {}
    for name, sign in (("positive", 1), ("negative", -1)):
        raw = validate_surface_geometry(
            surface,
            volume_shape_zyx=volume_shape_zyx,
            voxel_size_zyx_um=(voxel_um,) * 3,
            normal_offsets_um=offsets_um,
            normal_sign=sign,
        ).to_dict()
        reports[name] = {**compact_full_geometry(raw), "pass": full_geometry_pass(raw)}
    return reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stored = Path(args.surface)
    self_path = Path(args.self_intersection_audit)
    self_audit = json.loads(self_path.read_text(encoding="utf-8"))
    if not (
        self_audit.get("stored_nonadjacent_self_intersection_pass") is True
        and self_audit.get("full_res_nonadjacent_clearance_bound_pass") is True
        and float(self_audit.get("certified_full_res_nonadjacent_clearance_lower_bound_voxels", 0)) > 0
    ):
        raise ValueError("stored/full-bound self-intersection audit does not pass")

    initial = load_tifxyz_asset(
        stored,
        resolution="full",
        interpolation="linear",
        maximum_surface_pixels=args.maximum_full_pixels,
    )
    expected = derived_full_shape(initial.stored_shape, initial.scale_yx)
    initial_mask = initial.surface.valid
    quads = active_quads(initial_mask)
    render_mask = quad_union_vertex_mask(quads)
    if np.any(render_mask & ~initial_mask):
        raise RuntimeError("orphan cleanup added a full-resolution vertex")
    if not np.array_equal(active_quads(render_mask), quads):
        raise RuntimeError("orphan cleanup changed the full-resolution quad set")
    in_memory = SurfaceGrid.from_tifxyz(
        initial.surface.x,
        initial.surface.y,
        initial.surface.z,
        mask=render_mask,
    )
    in_memory_topology = scalable_topology_report(render_mask)
    in_memory_area = physical_area_cm2(in_memory, voxel_um=args.voxel_um)
    in_memory_geometry = geometry_reports(
        in_memory,
        voxel_um=args.voxel_um,
        volume_shape_zyx=args.volume_shape_zyx,
    )

    surface_dir = output / "full-resolution-surface"
    materialized = materialize_full(
        surface_dir,
        in_memory,
        candidate_id=args.candidate_id,
        voxel_um=args.voxel_um,
        parent=stored,
    )
    reloaded = load_tifxyz_asset(surface_dir, resolution="stored")
    if not np.array_equal(reloaded.surface.valid, render_mask):
        raise RuntimeError("float32 reload mask differs")
    reloaded_topology = scalable_topology_report(reloaded.surface.valid)
    reloaded_area = physical_area_cm2(reloaded.surface, voxel_um=args.voxel_um)
    reloaded_geometry = geometry_reports(
        reloaded.surface,
        voxel_um=args.voxel_um,
        volume_shape_zyx=args.volume_shape_zyx,
    )
    area_delta = reloaded_area - in_memory_area
    area_ceiling = max(1e-12, in_memory_area * 1e-4)
    gates = [
        {"name": "metadata_derived_shape", "pass": tuple(initial.output_shape) == expected,
         "value": list(initial.output_shape), "expected": list(expected)},
        {"name": "strictly_subtractive_orphan_cleanup", "pass": bool(np.all(~render_mask | initial_mask) and np.array_equal(active_quads(render_mask), quads)),
         "removed_vertices": int(np.count_nonzero(initial_mask & ~render_mask))},
        {"name": "in_memory_manifold_topology", "pass": not bool(in_memory_topology["nonmanifold_risk"])},
        {"name": "in_memory_area", "pass": in_memory_area >= args.minimum_area_cm2,
         "value_cm2": in_memory_area, "threshold_cm2": args.minimum_area_cm2},
        {"name": "in_memory_both_sign_native_qc", "pass": bool(in_memory_geometry["positive"]["pass"] and in_memory_geometry["negative"]["pass"])},
        {"name": "float32_reload_mask_identity", "pass": bool(np.array_equal(reloaded.surface.valid, render_mask))},
        {"name": "float32_reload_area_delta", "pass": abs(area_delta) <= area_ceiling,
         "delta_cm2": area_delta, "ceiling_cm2": area_ceiling},
        {"name": "float32_reload_manifold_topology", "pass": not bool(reloaded_topology["nonmanifold_risk"])},
        {"name": "float32_reload_area", "pass": reloaded_area >= args.minimum_area_cm2,
         "value_cm2": reloaded_area, "threshold_cm2": args.minimum_area_cm2},
        {"name": "float32_reload_both_sign_native_qc", "pass": bool(reloaded_geometry["positive"]["pass"] and reloaded_geometry["negative"]["pass"])},
        {"name": "stored_exact_nonadjacent_intersections_zero", "pass": self_audit["stored_nonadjacent_self_intersection_pass"] is True},
        {"name": "full_nonadjacent_clearance_bound_positive", "pass": self_audit["full_res_nonadjacent_clearance_bound_pass"] is True,
         "lower_bound_voxels": self_audit["certified_full_res_nonadjacent_clearance_lower_bound_voxels"]},
    ]
    passed = all(bool(gate["pass"]) for gate in gates)
    report = {
        "schema_version": 1,
        "status": "complete",
        "candidate_id": args.candidate_id,
        "verdict": "pass_to_raw_ct_planning" if passed else "stop_before_raw_ct",
        "go": passed,
        "paid_compute_cost_usd": 0.0,
        "raw_ct_network_bytes": 0,
        "inputs": {
            "stored_tifxyz": str(stored.resolve()),
            "stored_manifest": load_tifxyz_asset(stored, resolution="stored").manifest(),
            "self_intersection_audit": {"path": str(self_path.resolve()), "sha256": sha256_file(self_path)},
            "voxel_um": args.voxel_um,
            "volume_shape_zyx": list(args.volume_shape_zyx),
        },
        "resampling": {
            "method": "canonical mask-aware linear TIFFXYZ",
            "stored_shape": list(initial.stored_shape),
            "scale_yx": list(initial.scale_yx),
            "expected_full_shape": list(expected),
            "output_shape": list(initial.output_shape),
        },
        "orphan_cleanup": {
            "initial_valid_vertices": int(np.count_nonzero(initial_mask)),
            "final_valid_vertices": int(np.count_nonzero(render_mask)),
            "removed_unused_vertices": int(np.count_nonzero(initial_mask & ~render_mask)),
            "quad_count": int(np.count_nonzero(quads)),
        },
        "in_memory": {
            "mask_content_sha256": _array_sha256(render_mask),
            "area_cm2": in_memory_area,
            "topology": in_memory_topology,
            "native_geometry_by_sign": in_memory_geometry,
        },
        "float32_reload": {
            "surface": materialized,
            "mask_content_sha256": _array_sha256(reloaded.surface.valid),
            "points_content_sha256": _array_sha256(reloaded.surface.points_xyz),
            "area_cm2": reloaded_area,
            "area_delta_from_prewrite_cm2": area_delta,
            "topology": reloaded_topology,
            "native_geometry_by_sign": reloaded_geometry,
        },
        "self_intersection_scope": {
            "stored_nonadjacent_intersection_count": self_audit["nonadjacent_intersection_count"],
            "certified_full_nonadjacent_clearance_lower_bound_voxels": self_audit["certified_full_res_nonadjacent_clearance_lower_bound_voxels"],
            "local_full_raster_folds_and_topology": "covered by both-sign native QC and manifold gate",
            "shared_source_vertex_pairs": "treated as local adjacency, not nonadjacent intersections",
        },
        "gate_results": gates,
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "full-resolution-preflight.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate-id", required=True)
    value.add_argument("--surface", type=Path, required=True)
    value.add_argument("--self-intersection-audit", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--voxel-um", type=float, required=True)
    value.add_argument("--volume-shape-zyx", type=int, nargs=3, required=True)
    value.add_argument("--minimum-area-cm2", type=float, default=0.5)
    value.add_argument("--maximum-full-pixels", type=int, default=10_000_000)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        failure = {"schema_version": 1, "status": "error", "candidate_id": args.candidate_id,
                   "verdict": "validation_error", "error_type": type(error).__name__,
                   "error": str(error), "paid_compute_cost_usd": 0.0}
        _atomic_json(Path(args.output) / "full-resolution-preflight.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps({"verdict": report["verdict"], "go": report["go"],
                      "area_cm2": report["float32_reload"]["area_cm2"],
                      "output": str(Path(args.output) / "full-resolution-preflight.json")}, sort_keys=True))
    return 0 if report["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
