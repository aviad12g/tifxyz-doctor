#!/usr/bin/env python3
"""Generate sealed synthetic test cells and cache only preregistered rays."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cache_real_predictions import (
    RUNS,
    SOURCE_CHECKPOINT_SHA256,
    require_test_gate,
    run_identity,
    sha256_file,
)
from fusion_ray_readout import CENTER, K, sample_rays, verify_reference_reader
from inference import predict_volume
from normalization import validate_ct_contract


PAINTER_SHA256 = "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8"
TEST_SEEDS = (300, 301, 302, 303, 304)
PITCHES = (170.0, 200.0, 230.0, 260.0)
PAPYRUS_LEVELS = (35, 50, 65, 90)
CONTROL_PITCH = 700.0


def synthetic_cells() -> list[dict]:
    cells = []
    for seed in TEST_SEEDS:
        for pitch in PITCHES:
            for papyrus in PAPYRUS_LEVELS:
                cells.append(
                    {
                        "kind": "primary",
                        "seed": seed,
                        "pitch_um": pitch,
                        "papyrus": papyrus,
                        "kollesis": True,
                        "name": f"primary_seed{seed}_pitch{int(pitch)}_pap{papyrus}",
                    }
                )
        for papyrus in PAPYRUS_LEVELS:
            cells.append(
                {
                    "kind": "single_sheet_control",
                    "seed": seed,
                    "pitch_um": CONTROL_PITCH,
                    "papyrus": papyrus,
                    "kollesis": False,
                    "name": f"control_seed{seed}_pitch700_pap{papyrus}",
                }
            )
    return cells


def validate_ray_cache(path: Path) -> None:
    with np.load(path) as data:
        if set(data.files) != {"ray_probability", "ray_turn"}:
            raise RuntimeError(f"{path}: wrong ray-cache keys")
        probability = np.asarray(data["ray_probability"])
        turn = np.asarray(data["ray_turn"])
        if probability.shape != turn.shape or probability.ndim != 2:
            raise RuntimeError(f"{path}: ray-cache shape mismatch")
        if probability.shape[1] != K or not 0 < len(probability) <= 20_000:
            raise RuntimeError(f"{path}: invalid sampled-ray count/width")
        if probability.dtype != np.float16 or turn.dtype != np.int16:
            raise RuntimeError(f"{path}: ray-cache dtype mismatch")
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise RuntimeError(f"{path}: invalid ray probabilities")
        if np.any(turn < 0) or np.any(turn[:, CENTER] <= 0):
            raise RuntimeError(f"{path}: invalid ray instance labels")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=RUNS, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--painter-scripts", type=Path, required=True)
    parser.add_argument("--diagnostic-scripts", type=Path, required=True)
    parser.add_argument("--reference-reader", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--model-state", type=Path)
    parser.add_argument("--threshold-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.shard_count != 10:
        raise RuntimeError("the frozen synthetic test uses exactly 10 shards")
    if sha256_file(args.source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    painter = args.painter_scripts / "contrast_phantom.py"
    if sha256_file(painter) != PAINTER_SHA256:
        raise RuntimeError("finite-thickness painter SHA-256 mismatch")
    verify_reference_reader(args.reference_reader)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    selected_threshold = require_test_gate(
        args.threshold_manifest, args.run, split["records_sha256"]
    )

    arm, seed = run_identity(args.run)
    if (args.model_state is None) != (args.run == "baseline"):
        raise RuntimeError("baseline takes no model state; fine-tuned runs require one")
    sys.path.insert(0, str(args.diagnostic_scripts))
    import loader059

    loader059.CKPT = str(args.source_checkpoint)
    model, normalization, properties = loader059.load_059()
    ct_properties = validate_ct_contract(normalization, properties)
    model_state_sha256 = None
    if args.model_state is not None:
        state = torch.load(args.model_state, map_location="cpu", weights_only=False)
        if state.get("arm") != arm or int(state.get("seed", -1)) != seed:
            raise RuntimeError(f"model state identity does not match {args.run}")
        if state.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("fine-tuned state points to another source checkpoint")
        validate_ct_contract(
            state.get("normalization_scheme"), state.get("intensity_properties")
        )
        model.load_state_dict(state["model"], strict=True)
        model_state_sha256 = sha256_file(args.model_state)
    model.cuda().eval()

    sys.path.insert(0, str(args.painter_scripts))
    import contrast_phantom

    all_cells = synthetic_cells()
    cells = [
        cell for index, cell in enumerate(all_cells) if index % args.shard_count == args.shard_index
    ]
    if len(all_cells) != 100 or len(cells) != 10:
        raise RuntimeError("frozen synthetic grid/shard cardinality changed")
    output = args.out / args.run
    output.mkdir(parents=True, exist_ok=True)
    emitted = []
    for index, cell in enumerate(cells, start=1):
        path = output / f"{cell['name']}.npz"
        if not path.exists():
            volume, gt_surface, _, meta = contrast_phantom.emit_cell(
                12,
                cell["papyrus"],
                cell["pitch_um"],
                6.0,
                seed=cell["seed"],
                z_window_mm=5.0,
                voxel_um=30.0,
                sheet_um=150.0,
                arm="physical",
                kollesis=cell["kollesis"],
            )
            turn = np.asarray(meta["turn_id"], dtype=np.int16)
            if turn.ndim == 2:
                turn = np.broadcast_to(turn, volume.shape)
            turn = np.asarray(turn, dtype=np.int16)
            gt_surface = np.asarray(gt_surface, dtype=np.uint8)
            if volume.shape != gt_surface.shape or turn.shape != volume.shape:
                raise RuntimeError(f"{cell['name']}: painter shape mismatch")
            probability = predict_volume(
                model, np.asarray(volume), ct_properties, torch.device("cuda")
            )
            ray_probability, ray_turn = sample_rays(probability, gt_surface, turn)
            np.savez_compressed(
                path,
                ray_probability=ray_probability,
                ray_turn=ray_turn,
            )
        validate_ray_cache(path)
        emitted.append(
            {
                **cell,
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
        print(
            f"SYNTHETIC_RAYS {index}/{len(cells)} run={args.run} "
            f"shard={args.shard_index}/10 cell={cell['name']}",
            flush=True,
        )

    payload = {
        "schema_version": "1.0",
        "status": "sealed synthetic rays cached; endpoints not scored",
        "run": args.run,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_split_records_sha256": split["records_sha256"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "model_state_sha256": model_state_sha256,
        "selected_threshold": selected_threshold,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
        "cells": emitted,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest = args.out / f"synthetic_manifest_{args.run}_shard{args.shard_index:02d}.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("payload SHA-256:", payload["payload_sha256"])
    print("SEALED_SYNTHETIC_RAYS_CACHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
