#!/usr/bin/env python3
"""Frozen control/gap-arm decoder fine-tuning for surface_recto_059_redo."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from fusion_loss import fusion_aware_surface_loss
from gap_supervision import broadcast_gap_mask, inter_sheet_gap_mask
from normalization import ct_normalize, validate_ct_contract


PATCH = 160
ACCUM = 2
STEPS = 1500
LR = 5e-5
WEIGHT_DECAY = 1e-5
REAL_PER_SYNTHETIC = 3
TRAIN_SEEDS = tuple(range(100, 116))
PITCHES = (170.0, 200.0, 230.0, 260.0)
PAPYRUS = (35, 50, 65, 90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_real_data(
    root: Path, split_manifest: Path
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Resolve the already-frozen Scroll-1 split; never recompute it in training."""
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    records = [record for record in split["records"] if record["scroll"] == "s1"]
    images_by_name = {path.name: path for path in root.rglob("*_0000.tif")}
    labels_by_name = {
        p.name: p
        for p in root.rglob("*.tif")
        if not p.name.endswith("_0000.tif")
    }
    train, validation = [], []
    for record in records:
        image = images_by_name.get(record["image"])
        label = labels_by_name.get(record["label"])
        if image is None or label is None:
            raise RuntimeError(f"missing frozen pair for {record['image']}")
        if sha256_file(image) != record["image_sha256"]:
            raise RuntimeError(f"image hash mismatch for {record['image']}")
        if sha256_file(label) != record["label_sha256"]:
            raise RuntimeError(f"label hash mismatch for {record['label']}")
        if record["split"] not in {"train", "validation"}:
            raise RuntimeError(f"invalid Scroll-1 split for {record['image']}")
        destination = validation if record["split"] == "validation" else train
        destination.append((image, label))
    if len(train) < 100 or len(validation) != 24:
        raise RuntimeError(
            f"insufficient Scroll-1 split: train={len(train)} validation={len(validation)}"
        )
    return train, validation


def generate_synthetic_cache(cache: Path, painter_scripts: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(painter_scripts))
    import contrast_phantom

    paths = []
    combinations = [(pitch, papyrus) for pitch in PITCHES for papyrus in PAPYRUS]
    for seed, (pitch, papyrus) in zip(TRAIN_SEEDS, combinations):
        path = cache / f"seed{seed}_pitch{int(pitch)}_pap{papyrus}.npz"
        paths.append(path)
        if path.exists():
            continue
        print(f"SYNTH_GENERATE seed={seed} pitch={pitch:g} papyrus={papyrus}", flush=True)
        volume, surface, _, meta = contrast_phantom.emit_cell(
            12,
            papyrus,
            pitch,
            6.0,
            seed=seed,
            z_window_mm=5.0,
            voxel_um=30.0,
            sheet_um=150.0,
            arm="physical",
            kollesis=(seed % 2 == 0),
        )
        turn = np.asarray(meta["turn_id"], dtype=np.int16)
        gap_2d = inter_sheet_gap_mask(turn, surface[0], radius=3)
        gap = broadcast_gap_mask(gap_2d, volume.shape[0])
        if not gap.any() or np.any(gap & (surface > 0)):
            raise RuntimeError(f"invalid gap supervision for {path.name}")
        np.savez_compressed(
            path,
            volume=np.asarray(volume, dtype=np.uint8),
            target=np.asarray(surface, dtype=np.uint8),
            gap=gap,
            turn_id=turn,
        )
    return paths


def crop_arrays(arrays: tuple[np.ndarray, ...], rng: np.random.Generator):
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("paired arrays have different shapes")
    if any(length < PATCH for length in shape):
        raise ValueError(f"volume {shape} is smaller than {PATCH} cubed")
    starts = [int(rng.integers(0, length - PATCH + 1)) for length in shape]
    z, y, x = starts
    sl = np.s_[z : z + PATCH, y : y + PATCH, x : x + PATCH]
    cropped = tuple(np.ascontiguousarray(array[sl]) for array in arrays)
    for axis in range(3):
        if rng.random() < 0.5:
            cropped = tuple(np.ascontiguousarray(np.flip(array, axis=axis)) for array in cropped)
    return cropped


def load_real_crop(pair: tuple[Path, Path], rng: np.random.Generator):
    image = tifffile.imread(pair[0])
    target = tifffile.imread(pair[1])
    image, target = crop_arrays((image, target), rng)
    return image, (target > 0).astype(np.uint8), np.zeros_like(target, dtype=bool)


def load_synthetic_crop(path: Path, rng: np.random.Generator):
    with np.load(path) as data:
        return crop_arrays((data["volume"], data["target"], data["gap"]), rng)


def extract_logits(output):
    if isinstance(output, dict):
        output = output["surface"]
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim != 5 or output.shape[1] != 2:
        raise RuntimeError(f"unexpected model output shape {tuple(output.shape)}")
    return output


def configure_decoder_only(model: torch.nn.Module) -> int:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("shared_encoder.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if trainable == 0 or frozen == 0:
        raise RuntimeError(f"decoder-only partition failed: trainable={trainable} frozen={frozen}")
    print(f"PARAMETERS trainable={trainable:,} frozen={frozen:,}", flush=True)
    return trainable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "gap8"), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 23, 47), required=True)
    parser.add_argument("--real-data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--painter-scripts", type=Path, required=True)
    parser.add_argument("--diagnostic-scripts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(args.seed)

    train_pairs, validation_pairs = pair_real_data(args.real_data, args.split_manifest)
    synthetic = generate_synthetic_cache(args.work / "synthetic_train", args.painter_scripts)
    print(
        f"DATA real_train={len(train_pairs)} real_validation={len(validation_pairs)} "
        f"synthetic={len(synthetic)}",
        flush=True,
    )

    sys.path.insert(0, str(args.diagnostic_scripts))
    import loader059

    loader059.CKPT = str(args.checkpoint)
    model, normalization, properties = loader059.load_059()
    ct_properties = validate_ct_contract(normalization, properties)
    print(
        "NORMALIZATION scheme=CTNormalization "
        f"clip=[{ct_properties['percentile_00_5']:g},"
        f"{ct_properties['percentile_99_5']:g}] "
        f"mean={ct_properties['mean']:.6f} std={ct_properties['std']:.6f}",
        flush=True,
    )
    model.train()
    trainable = configure_decoder_only(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    gap_weight = 1.0 if args.arm == "control" else 8.0

    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    rolling = []
    for step in range(STEPS):
        for micro in range(ACCUM):
            sample_index = step * ACCUM + micro
            use_synthetic = sample_index % (REAL_PER_SYNTHETIC + 1) == REAL_PER_SYNTHETIC
            if use_synthetic:
                image, target, gap = load_synthetic_crop(
                    synthetic[int(rng.integers(0, len(synthetic)))], rng
                )
            else:
                image, target, gap = load_real_crop(
                    train_pairs[int(rng.integers(0, len(train_pairs)))], rng
                )
            x = torch.from_numpy(ct_normalize(image, ct_properties))[None, None].cuda()
            y = torch.from_numpy(target.astype(np.int64))[None].cuda()
            g = torch.from_numpy(gap.astype(bool))[None].cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = extract_logits(model(x))
                loss, report = fusion_aware_surface_loss(
                    logits, y, g, gap_weight=gap_weight
                )
                loss = loss / ACCUM
            scaler.scale(loss).backward()
            rolling.append(float(loss.detach()) * ACCUM)
            del x, y, g, logits, loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 12.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if (step + 1) % 50 == 0:
            print(
                f"TRAIN arm={args.arm} seed={args.seed} step={step+1}/{STEPS} "
                f"loss={np.mean(rolling[-100:]):.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} "
                f"elapsed={time.monotonic()-start:.1f}s",
                flush=True,
            )

    output = args.work / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"surface059_{args.arm}_seed{args.seed}.pth"
    payload = {
        "model": model.state_dict(),
        "arm": args.arm,
        "seed": args.seed,
        "steps": STEPS,
        "gap_weight": gap_weight,
        "trainable_parameters": trainable,
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "real_validation_files": [p[0].name for p in validation_pairs],
    }
    torch.save(payload, checkpoint_path)
    metadata = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "arm": args.arm,
        "seed": args.seed,
        "steps": STEPS,
        "gap_weight": gap_weight,
        "elapsed_seconds": time.monotonic() - start,
        "trainable_parameters": trainable,
        "real_train_count": len(train_pairs),
        "real_validation_count": len(validation_pairs),
        "synthetic_count": len(synthetic),
        "source_checkpoint_sha256": sha256_file(args.checkpoint),
        "normalization_scheme": "CTNormalization",
        "intensity_properties": ct_properties,
    }
    metadata_path = checkpoint_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    print("FUSION_AWARE_TRAINING_COMPLETE", flush=True)
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
