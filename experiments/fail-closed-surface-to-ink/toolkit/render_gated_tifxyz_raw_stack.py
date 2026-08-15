#!/usr/bin/env python3
"""Plan and render a fully gated 93-layer TIFFXYZ surface from public raw CT."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import render_pherc1203_auto_grown_raw_stack as engine
from sample_m7_seed_chunk import DEFAULT_BLOSC
from sample_raw_ct_seed_cube import sha256_file
from tifxyz_render_pipeline import RenderOptions, load_tifxyz_asset, render_surface_to_directory
from validate_public_m7_patch import _array_sha256, _atomic_json


def require_inputs(args: argparse.Namespace, asset: Any) -> dict[str, Any]:
    full_path = Path(args.full_preflight)
    full = json.loads(full_path.read_text(encoding="utf-8"))
    if full.get("go") is not True or full.get("verdict") != "pass_to_raw_ct_planning":
        raise ValueError("full-resolution preflight does not authorize raw planning")
    expected_mask = full["float32_reload"]["mask_content_sha256"]
    expected_points = full["float32_reload"]["points_content_sha256"]
    if _array_sha256(asset.surface.valid) != expected_mask:
        raise ValueError("full surface mask differs from gated preflight")
    if _array_sha256(asset.surface.points_xyz) != expected_points:
        raise ValueError("full surface coordinates differ from gated float32 reload")
    self_path = Path(args.self_intersection_audit)
    self_audit = json.loads(self_path.read_text(encoding="utf-8"))
    if not (
        self_audit.get("stored_nonadjacent_self_intersection_pass") is True
        and self_audit.get("full_res_nonadjacent_clearance_bound_pass") is True
        and float(self_audit.get("certified_full_res_nonadjacent_clearance_lower_bound_voxels", 0)) > 0
    ):
        raise ValueError("self-intersection evidence does not authorize raw diagnostic")
    return {
        "full_preflight": {"path": str(full_path.resolve()), "sha256": sha256_file(full_path)},
        "self_intersection_audit": {"path": str(self_path.resolve()), "sha256": sha256_file(self_path)},
        "surface_manifest": asset.manifest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    asset = load_tifxyz_asset(Path(args.surface), resolution="stored")
    evidence = require_inputs(args, asset)

    shape = tuple(int(value) for value in args.volume_shape_zyx)
    engine.CANDIDATE_ID = args.candidate_id
    engine.PINNED_VOLUME_SHAPE_ZYX = shape
    engine.VOXEL_UM = float(args.voxel_um)
    engine.ALGORITHM_VERSION = "generic-gated-free-raw-render-v1"
    engine.RAW_USER_AGENT = "vesuvius-first-letters-gated-free-raw-render/1"

    spec, metadata = engine.validate_raw_metadata(
        args.raw_root, args.array_path, output / "raw-metadata-cache"
    )
    plan = engine.build_chunk_plan(
        asset, spec, args.raw_root, args.array_path, tile_size=args.tile_size
    )
    plan.update(
        {
            "candidate_id": args.candidate_id,
            "algorithm_version": engine.ALGORITHM_VERSION,
            "voxel_um": args.voxel_um,
            "gated_evidence": evidence,
        }
    )
    plan_path = output / "raw-chunk-plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous != plan:
            raise RuntimeError("immutable raw chunk plan differs from existing plan")
    else:
        _atomic_json(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    free_bytes = shutil.disk_usage(output).free
    precommit = {
        "path": str(plan_path.resolve()),
        "sha256": plan_sha,
        "written_before_raw_chunk_access": True,
        "free_disk_bytes_at_plan_commit": free_bytes,
        "unique_chunk_count": plan["unique_chunk_count"],
        "decoded_chunk_bytes_upper_bound": plan["decoded_chunk_bytes_upper_bound"],
    }
    precommit_path = output / "raw-access-precommit.json"
    if precommit_path.exists():
        prior = json.loads(precommit_path.read_text(encoding="utf-8"))
        for key in ("path", "sha256", "written_before_raw_chunk_access", "unique_chunk_count", "decoded_chunk_bytes_upper_bound"):
            if prior.get(key) != precommit.get(key):
                raise RuntimeError("raw access precommit changed")
        precommit = prior
    else:
        _atomic_json(precommit_path, precommit)
    if args.plan_only:
        return {
            "schema_version": 1,
            "status": "plan_complete",
            "verdict": "ready_for_free_raw_render",
            "paid_compute_cost_usd": 0.0,
            "chunk_plan": precommit,
        }

    blosc = Path(args.blosc)
    if not blosc.is_file():
        raise FileNotFoundError("pinned Blosc codec is missing")
    cache = Path(args.raw_cache)
    prefetch = engine.prefetch_planned_chunks(
        {**plan, "plan_sha256": plan_sha}, spec, cache, blosc,
        workers=args.prefetch_workers,
    )
    prefetch_path = output / "raw-prefetch-manifest.json"
    _atomic_json(prefetch_path, prefetch)
    allowed = {
        tuple(int(value) for value in item["chunk_index_zyx"])
        for item in plan["unique_chunks"]
    }
    volume = engine.PlannedHashedPublicZarrArray(
        root_url=args.raw_root,
        array_path=args.array_path,
        spec=spec,
        cache_directory=cache,
        blosc_path=blosc,
        decoded_lru_chunks=args.decoded_lru_chunks,
        allowed_chunks=allowed,
    )
    surface_manifest = {
        "candidate_id": args.candidate_id,
        "asset": asset.manifest(),
        "gated_evidence": evidence,
    }

    sparse_stack = output / "raw-sparse-seven"
    sparse = RenderOptions(
        voxel_size_zyx_um=(args.voxel_um,) * 3,
        offsets_um=tuple(float(value) * args.voxel_um for value in engine.SPARSE_OFFSETS),
        signs=("positive",),
        tile_shape=(args.tile_size, args.tile_size),
        output_dtype="uint8",
        fill_value=0.0,
        png_mode="preview",
        overwrite=args.overwrite,
        hash_outputs=True,
    )
    sparse_manifest = render_surface_to_directory(
        volume, asset.surface, sparse_stack, options=sparse,
        tifxyz_manifest=surface_manifest, volume_manifest=metadata,
        progress=lambda index, total, detail: print(json.dumps({"stage": "sparse", "tile": index, "tiles": total, **detail}), flush=True),
    )
    early = engine.sparse_diagnostics(sparse_stack, output, asset.surface.valid)
    _atomic_json(output / "early-diagnostics.json", early)

    full_stack = output / "raw-surface-stack"
    full = RenderOptions(
        voxel_size_zyx_um=(args.voxel_um,) * 3,
        offsets_um=tuple(float(value) * args.voxel_um for value in engine.FULL_OFFSETS),
        signs=("positive",),
        tile_shape=(args.tile_size, args.tile_size),
        output_dtype="uint8",
        fill_value=0.0,
        png_mode="preview",
        overwrite=args.overwrite,
        hash_outputs=True,
    )
    full_manifest = render_surface_to_directory(
        volume, asset.surface, full_stack, options=full,
        tifxyz_manifest=surface_manifest, volume_manifest=metadata,
        progress=lambda index, total, detail: print(json.dumps({"stage": "full", "tile": index, "tiles": total, **detail}), flush=True),
    )
    verification = engine.verify_full_stack(full_stack, asset.surface.valid)
    diagnostics = engine.build_diagnostics(full_stack, output)
    reader = volume.manifest()
    if set(volume.records) - allowed:
        raise RuntimeError("raw reader accessed an unplanned chunk")
    mappings = engine.model_window_mappings()
    mappings["stored_frame_to_offset_micrometers"] = {
        str(index): float(value) * args.voxel_um
        for index, value in enumerate(engine.FULL_OFFSETS)
    }
    final = {
        "schema_version": 1,
        "algorithm_version": engine.ALGORITHM_VERSION,
        "status": "complete",
        "verdict": "model_ready_free_raw_stack",
        "candidate_id": args.candidate_id,
        "paid_compute_cost_usd": 0.0,
        "gated_evidence": evidence,
        "raw_public_zarr": {
            "metadata": metadata,
            "chunk_plan": precommit,
            "prefetch": {"path": str(prefetch_path.resolve()), "sha256": sha256_file(prefetch_path),
                         "record_count": prefetch["record_count"], "network_fetch_count": prefetch["network_fetch_count"],
                         "network_payload_bytes": prefetch["network_payload_bytes"]},
            "chunk_reader": reader,
            "codec_path": str(blosc.resolve()),
            "codec_sha256": sha256_file(blosc),
        },
        "sparse_render": {"offsets_voxels": list(engine.SPARSE_OFFSETS), "renderer": sparse_manifest, "early_diagnostics": early},
        "raw_stack": {**verification, "directory": str((full_stack / "positive").resolve()),
                      "offsets_voxels": list(engine.FULL_OFFSETS),
                      "offsets_micrometers": [float(value) * args.voxel_um for value in engine.FULL_OFFSETS],
                      "renderer": full_manifest},
        "sign_order_and_model_window_mappings": mappings,
        "diagnostics": diagnostics,
        "limitations": [
            "Raw morphology is not a semantic claim of letters.",
            "No detector, OCR, or ink model was run.",
            "Only the positive-normal stack is stored; negative is exact layer reversal.",
        ],
    }
    final_path = output / "raw-stack-manifest.json"
    _atomic_json(final_path, final)
    manifest = engine.write_manifest(output)
    return {**final, "artifact_manifest": manifest}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate-id", required=True)
    value.add_argument("--surface", type=Path, required=True)
    value.add_argument("--full-preflight", type=Path, required=True)
    value.add_argument("--self-intersection-audit", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--raw-root", required=True)
    value.add_argument("--array-path", default="0")
    value.add_argument("--raw-cache", type=Path, required=True)
    value.add_argument("--blosc", type=Path, default=DEFAULT_BLOSC)
    value.add_argument("--voxel-um", type=float, required=True)
    value.add_argument("--volume-shape-zyx", type=int, nargs=3, required=True)
    value.add_argument("--tile-size", type=int, default=256)
    value.add_argument("--decoded-lru-chunks", type=int, default=32)
    value.add_argument("--prefetch-workers", type=int, default=8)
    value.add_argument("--plan-only", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        failure = {"schema_version": 1, "status": "error", "candidate_id": args.candidate_id,
                   "verdict": "render_error", "error_type": type(error).__name__,
                   "error": str(error), "paid_compute_cost_usd": 0.0}
        _atomic_json(Path(args.output) / "raw-render-error.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "verdict": result["verdict"],
                      "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
