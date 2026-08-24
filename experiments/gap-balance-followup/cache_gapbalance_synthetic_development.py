"""Cache only frozen GapBalance synthetic-development rays, without scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gapbalance_development as contract
import numpy as np
import torch

FUSION_ROOT = Path(__file__).resolve().parents[1] / "fusion-aware-surface"
sys.path.insert(0, str(FUSION_ROOT))
from fusion_ray_readout import CENTER, K, sample_rays, verify_reference_reader
from inference import predict_volume
from normalization import validate_ct_contract

SOURCE_CHECKPOINT_SHA256 = "f1990a02ac91889c1f989522ae0e45421a91cb666320448aaf579d42b081636f"
SPLIT_MANIFEST_SHA256 = "dedc881134d9de2ed2605162f82dfb219b52c6e05629c68223100b148b15d4fe"
SPLIT_RECORDS_SHA256 = "20c600d6061bf8715ada20423a05c2f03a9bc14827b2c79bac7b7e1b3cc0499d"
PAINTER_SHA256 = "41f2a097b819bc486819d6702ed3949d0bdfe722c2405b1fe86828062da1d9b8"
STATUS = "GapBalance synthetic development rays cached; endpoints not scored"
SHARD_COUNT = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if (
            not np.isfinite(probability).all()
            or probability.min() < 0
            or probability.max() > 1
        ):
            raise RuntimeError(f"{path}: invalid ray probabilities")
        if np.any(turn < 0) or np.any(turn[:, CENTER] <= 0):
            raise RuntimeError(f"{path}: invalid ray instance labels")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=contract.RUNS, required=True)
    parser.add_argument("--shard-index", type=int, choices=range(SHARD_COUNT), required=True)
    parser.add_argument("--painter-scripts", type=Path, required=True)
    parser.add_argument("--diagnostic-scripts", type=Path, required=True)
    parser.add_argument("--reference-reader", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--model-state", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    arm, seed = contract.run_identity(args.run)
    if sha256_file(args.source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    if sha256_file(args.split_manifest) != SPLIT_MANIFEST_SHA256:
        raise RuntimeError("Scroll-1 split manifest SHA-256 mismatch")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split.get("records_sha256") != SPLIT_RECORDS_SHA256:
        raise RuntimeError("Scroll-1 split records SHA-256 mismatch")
    painter = args.painter_scripts / "contrast_phantom.py"
    if sha256_file(painter) != PAINTER_SHA256:
        raise RuntimeError("finite-thickness painter SHA-256 mismatch")
    verify_reference_reader(args.reference_reader)

    sys.path.insert(0, str(args.diagnostic_scripts))
    import loader059

    loader059.CKPT = str(args.source_checkpoint)
    model, normalization, properties = loader059.load_059()
    ct_properties = validate_ct_contract(normalization, properties)
    state = torch.load(args.model_state, map_location="cpu", weights_only=False)
    if state.get("arm") != arm or int(state.get("seed", -1)) != seed:
        raise RuntimeError(f"model state identity does not match {args.run}")
    if state.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("fine-tuned state points to another source checkpoint")
    validate_ct_contract(
        state.get("normalization_scheme"), state.get("intensity_properties")
    )
    model.load_state_dict(state["model"], strict=True)
    model.cuda().eval()
    model_state_sha256 = sha256_file(args.model_state)

    sys.path.insert(0, str(args.painter_scripts))
    import contrast_phantom

    all_cells = contract.synthetic_cells()
    cells = [
        cell
        for index, cell in enumerate(all_cells)
        if index % SHARD_COUNT == args.shard_index
    ]
    if len(cells) != 20:
        raise RuntimeError("frozen synthetic development shard is not exactly 20 cells")
    output = args.out / args.run
    output.mkdir(parents=True, exist_ok=True)
    emitted = []
    for index, cell in enumerate(cells, start=1):
        cache = output / f"{cell['name']}.npz"
        if not cache.exists():
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
                cache, ray_probability=ray_probability, ray_turn=ray_turn
            )
        validate_ray_cache(cache)
        emitted.append(
            {
                **cell,
                "file": cache.name,
                "sha256": sha256_file(cache),
                "bytes": cache.stat().st_size,
            }
        )
        print(
            f"GAPBALANCE_SYNTHETIC_DEVELOPMENT {index}/20 run={args.run} "
            f"shard={args.shard_index}/{SHARD_COUNT}",
            flush=True,
        )

    payload = {
        "schema_version": "1.0",
        "status": STATUS,
        "run": args.run,
        "shard_index": args.shard_index,
        "shard_count": SHARD_COUNT,
        "source_split_records_sha256": SPLIT_RECORDS_SHA256,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "model_state_sha256": model_state_sha256,
        "selected_threshold": None,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
        "cells": emitted,
        "confirmation_outputs_inspected": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest = (
        args.out
        / f"synthetic_development_manifest_{args.run}_shard{args.shard_index:02d}.json"
    )
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("payload SHA-256:", payload["payload_sha256"])
    print("GAPBALANCE_SYNTHETIC_DEVELOPMENT_CACHED_NOT_SCORED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
