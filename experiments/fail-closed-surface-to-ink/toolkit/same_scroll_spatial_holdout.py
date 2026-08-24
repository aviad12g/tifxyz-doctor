#!/usr/bin/env python3
"""Test whether coarse ink signal transfers across distant regions of one scroll.

This diagnostic intentionally does no checkpoint selection on the spatial
holdout.  It trains for a fixed number of epochs on one side of one labeled
scroll, evaluates that fitted model once on a separated region of the same
scroll, and leaves the PHerc841 cross-scroll holdout untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_coarse_ink_detector import (
    CoarseInkDataset,
    CoarseInkNet,
    _positive_weight,
    evaluate_orientation_pair,
    masked_bce_dice_loss,
    set_deterministic,
)


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialSplitConfig:
    group: str = "0500p2"
    train_max_y_index: int = 29
    holdout_min_y_index: int = 31
    seed: int = 5002026
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    base_channels: int = 12
    num_workers: int = 0
    minimum_train_auroc: float = 0.80
    minimum_train_ap_above_prevalence: float = 0.15
    minimum_holdout_auroc: float = 0.70
    minimum_holdout_ap_above_prevalence: float = 0.10
    minimum_holdout_f1: float = 0.40
    minimum_orientation_pearson: float = 0.75
    minimum_orientation_top5_jaccard: float = 0.25


def blocked_spatial_split(
    records: Iterable[dict[str, Any]], config: SpatialSplitConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    group_records = [
        record
        for record in records
        if record["group"] == config.group and record["split"] == "train"
    ]
    if not group_records:
        raise ValueError(f"no training records for group {config.group}")
    train = [
        record
        for record in group_records
        if int(record["y_index"]) <= config.train_max_y_index
    ]
    holdout = [
        record
        for record in group_records
        if int(record["y_index"]) >= config.holdout_min_y_index
    ]
    excluded = [
        record
        for record in group_records
        if config.train_max_y_index < int(record["y_index"]) < config.holdout_min_y_index
    ]
    if not train or not holdout:
        raise ValueError("both spatial regions must contain records")
    train_ids = {record["sample_id"] for record in train}
    holdout_ids = {record["sample_id"] for record in holdout}
    if train_ids & holdout_ids:
        raise ValueError("sample leakage across spatial split")
    max_train_y1 = max(int(record["y1"]) for record in train)
    min_holdout_y0 = min(int(record["y0"]) for record in holdout)
    gap_pixels = min_holdout_y0 - max_train_y1
    if gap_pixels <= 0:
        raise ValueError("spatial regions overlap or touch")
    voxel_sizes = {float(record["coarse_voxel_um"]) for record in group_records}
    if len(voxel_sizes) != 1:
        raise ValueError("group contains inconsistent coarse voxel sizes")
    voxel_um = voxel_sizes.pop()
    split_evidence = {
        "axis": "y",
        "train_rule": f"y_index <= {config.train_max_y_index}",
        "holdout_rule": f"y_index >= {config.holdout_min_y_index}",
        "excluded_buffer_indices": sorted(
            {int(record["y_index"]) for record in excluded}
        ),
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_max_y1_exclusive": max_train_y1,
        "holdout_min_y0": min_holdout_y0,
        "minimum_gap_pixels": gap_pixels,
        "minimum_gap_um": gap_pixels * voxel_um,
        "minimum_gap_mm": gap_pixels * voxel_um / 1000.0,
        "coarse_voxel_um": voxel_um,
        "train_sample_ids": sorted(train_ids),
        "holdout_sample_ids": sorted(holdout_ids),
        "sealed_cross_scroll_group": "841",
        "sealed_cross_scroll_group_used": False,
    }
    return train, holdout, split_evidence


def classify_verdict(
    train_gates: dict[str, bool], holdout_gates: dict[str, bool]
) -> str:
    if not all(train_gates.values()):
        return "training_fit_failed_inconclusive"
    if not all(holdout_gates.values()):
        return "same_scroll_spatial_transfer_failed"
    return "same_scroll_coarse_signal_recovered"


def run_experiment(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    config: SpatialSplitConfig,
    *,
    device_name: str,
) -> dict[str, Any]:
    set_deterministic(config.seed)
    manifest = json.loads(manifest_path.read_text())
    train_records, holdout_records, split_evidence = blocked_spatial_split(
        manifest["records"], config
    )
    if any(record["group"] == "841" for record in train_records + holdout_records):
        raise ValueError("sealed PHerc841 records entered the diagnostic")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train_dataset = CoarseInkDataset(
        train_records, root=dataset_root, augment=True, seed=config.seed
    )
    train_eval_dataset = CoarseInkDataset(
        train_records, root=dataset_root, augment=False, seed=config.seed
    )
    holdout_dataset = CoarseInkDataset(
        holdout_records, root=dataset_root, augment=False, seed=config.seed
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = CoarseInkNet(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )
    positive_weight = _positive_weight(train_records, dataset_root)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            volume = batch["volume"].to(device)
            label = batch["label"].to(device)
            supervision = batch["supervision"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(volume)
                loss = masked_bce_dice_loss(
                    logits, label, supervision, positive_weight=positive_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"epoch={epoch + 1} loss={record['training_loss']:.6f} "
            f"lr={record['learning_rate']:.8f}",
            flush=True,
        )

    # The spatial holdout is evaluated exactly once, after the fixed training run.
    train_final = evaluate_orientation_pair(model, train_eval_loader, device)
    holdout_final = evaluate_orientation_pair(model, holdout_loader, device)
    train_forward = train_final["forward"]
    holdout_forward = holdout_final["forward"]
    train_gates = {
        "auroc": train_forward["auroc"] >= config.minimum_train_auroc,
        "ap_above_prevalence": (
            train_forward["average_precision"] - train_forward["prevalence"]
            >= config.minimum_train_ap_above_prevalence
        ),
    }
    holdout_gates = {
        "auroc": holdout_forward["auroc"] >= config.minimum_holdout_auroc,
        "ap_above_prevalence": (
            holdout_forward["average_precision"] - holdout_forward["prevalence"]
            >= config.minimum_holdout_ap_above_prevalence
        ),
        "f1": holdout_forward["f1"] >= config.minimum_holdout_f1,
        "orientation_pearson": (
            holdout_final["orientation"]["pearson"]
            >= config.minimum_orientation_pearson
        ),
        "orientation_top5_jaccard": (
            holdout_final["orientation"]["top5_jaccard"]
            >= config.minimum_orientation_top5_jaccard
        ),
    }
    verdict = classify_verdict(train_gates, holdout_gates)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "fixed-epoch-model.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
            "base_channels": config.base_channels,
            "epochs": config.epochs,
            "manifest_sha256": sha256_file(manifest_path),
            "group": config.group,
        },
        checkpoint_path,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "same_scroll_blocked_spatial_holdout",
        "verdict": verdict,
        "config": asdict(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_root": str(dataset_root),
        "split_evidence": split_evidence,
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "positive_weight": positive_weight,
        "history": history,
        "train_final": train_final,
        "spatial_holdout_final": holdout_final,
        "train_gates": train_gates,
        "spatial_holdout_gates": holdout_gates,
        "spatial_holdout_evaluations": 1,
        "pherc841_loaded": False,
        "eligible_scroll_inference_started": False,
        "elapsed_seconds": time.time() - started,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = output_dir / "same-scroll-spatial-holdout-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SpatialSplitConfig(epochs=args.epochs, num_workers=args.num_workers)
    report = run_experiment(
        args.manifest,
        args.dataset_root,
        args.output,
        config,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "train_gates": report["train_gates"],
                "spatial_holdout_gates": report["spatial_holdout_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == "same_scroll_coarse_signal_recovered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
