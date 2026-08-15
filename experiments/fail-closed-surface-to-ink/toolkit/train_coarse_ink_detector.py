#!/usr/bin/env python3
"""Train and validate a compact ink detector in the ~9 um physical domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("volume must be ZYX")
    finite = np.isfinite(volume)
    signal = volume[finite & (volume > 0)]
    if signal.size < 32:
        raise ValueError("volume has too few finite positive voxels")
    low, high = np.percentile(signal, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("invalid normalization percentiles")
    normalized = np.clip((volume - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32, copy=False)


class CoarseInkDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        root: Path,
        augment: bool,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.root = root
        self.augment = augment
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, record: dict[str, Any]) -> Path:
        path = Path(record["npz_path"])
        if path.is_absolute():
            return path
        direct = self.root / path
        if direct.is_file():
            return direct
        # Payloads may place the dataset root beside the manifest rather than
        # preserving the original workspace prefix.
        marker = "coarse-ink-domain-v1"
        parts = path.parts
        if marker in parts:
            relative = Path(*parts[parts.index(marker) + 1 :])
            candidate = self.root / relative
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self._resolve_path(record)
        with np.load(path) as payload:
            volume = normalize_volume(payload["volume"])
            label = (payload["label"] > 0).astype(np.float32)
            supervision = (payload["supervision"] > 0).astype(np.float32)
        if volume.shape != (17, 128, 128):
            raise ValueError(f"unexpected volume shape {volume.shape} for {path}")
        if label.shape != (128, 128) or supervision.shape != (128, 128):
            raise ValueError(f"unexpected target shape for {path}")

        if self.augment:
            rng = np.random.default_rng(self.seed + index + random.randrange(1 << 30))
            if rng.random() < 0.5:
                volume = volume[::-1].copy()
            k = int(rng.integers(0, 4))
            volume = np.rot90(volume, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            supervision = np.rot90(supervision, k, axes=(0, 1)).copy()
            if rng.random() < 0.5:
                volume = volume[:, :, ::-1].copy()
                label = label[:, ::-1].copy()
                supervision = supervision[:, ::-1].copy()
            if rng.random() < 0.5:
                volume = volume[:, ::-1, :].copy()
                label = label[::-1, :].copy()
                supervision = supervision[::-1, :].copy()
            gain = float(rng.uniform(0.9, 1.1))
            offset = float(rng.uniform(-0.05, 0.05))
            noise = rng.normal(0.0, 0.015, size=volume.shape).astype(np.float32)
            volume = np.clip(volume * gain + offset + noise, 0.0, 1.0)

        return {
            "volume": torch.from_numpy(volume[None]),
            "label": torch.from_numpy(label[None]),
            "supervision": torch.from_numpy(supervision[None]),
            "sample_id": record["sample_id"],
            "group": record["group"],
        }


class ConvBlock3d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while channels % groups:
        groups -= 1
    return groups


class ProjectDepth(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(inputs, dim=2)
        maximum = torch.amax(inputs, dim=2)
        return self.project(torch.cat([mean, maximum], dim=1))


class DecoderBlock2d(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([inputs, skip], dim=1))


class CoarseInkNet(nn.Module):
    """A compact 3-D encoder with a 2-D U-Net decoder (about 1M parameters)."""

    def __init__(self, base_channels: int = 12) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc0 = ConvBlock3d(1, channels[0])
        self.enc1 = ConvBlock3d(channels[0], channels[1])
        self.enc2 = ConvBlock3d(channels[1], channels[2])
        self.enc3 = ConvBlock3d(channels[2], channels[3])
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.projections = nn.ModuleList(ProjectDepth(value) for value in channels)
        self.dec2 = DecoderBlock2d(channels[3], channels[2], channels[2])
        self.dec1 = DecoderBlock2d(channels[2], channels[1], channels[1])
        self.dec0 = DecoderBlock2d(channels[1], channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x0 = self.enc0(inputs)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        p0, p1, p2, p3 = (
            projection(features)
            for projection, features in zip(self.projections, (x0, x1, x2, x3))
        )
        decoded = self.dec2(p3, p2)
        decoded = self.dec1(decoded, p1)
        decoded = self.dec0(decoded, p0)
        return self.output(decoded)


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != supervision.shape:
        raise ValueError("logits, target, and supervision shapes must match")
    valid = supervision > 0.5
    if not torch.any(valid):
        raise ValueError("batch contains no supervised pixels")
    weights = torch.where(target > 0.5, positive_weight, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = torch.sum(bce[valid] * weights[valid]) / torch.sum(weights[valid])
    probability = torch.sigmoid(logits) * supervision
    target_masked = target * supervision
    intersection = torch.sum(probability * target_masked, dim=(1, 2, 3))
    denominator = torch.sum(probability, dim=(1, 2, 3)) + torch.sum(
        target_masked, dim=(1, 2, 3)
    )
    dice_loss = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce + dice_loss


def binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    precision = np.cumsum(sorted_target) / np.arange(1, target.size + 1)
    return float(np.sum(precision * sorted_target) / positives)


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.uint8).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    positive = int(np.count_nonzero(target))
    negative = int(target.size - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        end = start + 1
        while end < score.size and sorted_score[end] == sorted_score[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(np.sum(ranks[target > 0]))
    return (positive_rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def top_fraction_jaccard(first: np.ndarray, second: np.ndarray, fraction: float) -> float:
    if first.shape != second.shape:
        raise ValueError("score arrays must have the same shape")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    count = max(1, int(math.ceil(first.size * fraction)))
    first_indices = np.argpartition(first, -count)[-count:]
    second_indices = np.argpartition(second, -count)[-count:]
    intersection = len(np.intersect1d(first_indices, second_indices, assume_unique=False))
    union = len(np.union1d(first_indices, second_indices))
    return intersection / union


def calculate_metrics(target: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    target = np.asarray(target, dtype=np.uint8).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    prediction = score >= threshold
    truth = target > 0
    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "average_precision": binary_average_precision(target, score),
        "auroc": binary_auroc(target, score),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence": float(np.mean(truth)),
    }


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 12032026
    dev_group: str = "814"
    epochs: int = 35
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    base_channels: int = 12
    patience: int = 7
    num_workers: int = 2
    minimum_holdout_ap: float = 0.50
    minimum_holdout_auroc: float = 0.65
    minimum_holdout_f1: float = 0.40
    minimum_orientation_pearson: float = 0.75
    minimum_orientation_top5_jaccard: float = 0.25


def split_records(
    records: Iterable[dict[str, Any]], dev_group: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train, dev, holdout = [], [], []
    for record in records:
        if record["split"] == "holdout":
            holdout.append(record)
        elif record["group"] == dev_group:
            dev.append(record)
        else:
            train.append(record)
    if not train or not dev or not holdout:
        raise ValueError("train, dev, and holdout splits must all be non-empty")
    if {r["group"] for r in train} & {r["group"] for r in dev + holdout}:
        raise ValueError("group leakage detected")
    if {r["group"] for r in dev} & {r["group"] for r in holdout}:
        raise ValueError("group leakage detected")
    return train, dev, holdout


@torch.no_grad()
def infer_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    reverse_depth: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        volume = batch["volume"].to(device)
        if reverse_depth:
            volume = torch.flip(volume, dims=(2,))
        logits = model(volume)
        probability = torch.sigmoid(logits).cpu().numpy()
        target = batch["label"].numpy()
        supervision = batch["supervision"].numpy() > 0.5
        targets.append(target[supervision].astype(np.uint8))
        scores.append(probability[supervision].astype(np.float32))
        sample_ids.extend(batch["sample_id"])
    return np.concatenate(targets), np.concatenate(scores), sample_ids


def evaluate_orientation_pair(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    target_forward, forward, ids_forward = infer_loader(
        model, loader, device, reverse_depth=False
    )
    target_reverse, reverse, ids_reverse = infer_loader(
        model, loader, device, reverse_depth=True
    )
    if ids_forward != ids_reverse or not np.array_equal(target_forward, target_reverse):
        raise ValueError("orientation evaluations are not aligned")
    if np.std(forward) == 0 or np.std(reverse) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(forward, reverse)[0, 1])
    return {
        "forward": calculate_metrics(target_forward, forward),
        "reverse": calculate_metrics(target_reverse, reverse),
        "orientation": {
            "pearson": pearson,
            "mean_absolute_difference": float(np.mean(np.abs(forward - reverse))),
            "top5_jaccard": top_fraction_jaccard(forward, reverse, 0.05),
            "top1_jaccard": top_fraction_jaccard(forward, reverse, 0.01),
        },
        "supervised_pixel_count": int(target_forward.size),
        "sample_count": len(ids_forward),
        "sample_ids": ids_forward,
    }


def _positive_weight(records: Iterable[dict[str, Any]], root: Path) -> float:
    positive = 0
    valid = 0
    for record in records:
        path = Path(record["npz_path"])
        if not path.is_absolute() and not path.is_file():
            marker = "coarse-ink-domain-v1"
            parts = path.parts
            if marker in parts:
                path = root / Path(*parts[parts.index(marker) + 1 :])
            else:
                path = root / path
        with np.load(path) as payload:
            label = payload["label"] > 0
            supervision = payload["supervision"] > 0
        positive += int(np.count_nonzero(label & supervision))
        valid += int(np.count_nonzero(supervision))
    negative = valid - positive
    if positive == 0 or negative <= 0:
        raise ValueError("training labels must contain both classes")
    return float(min(8.0, max(1.0, negative / positive)))


def train_model(
    manifest_path: Path,
    dataset_root: Path,
    output_dir: Path,
    config: TrainingConfig,
    *,
    device_name: str,
    max_train_samples: int | None = None,
) -> dict[str, Any]:
    set_deterministic(config.seed)
    manifest = json.loads(manifest_path.read_text())
    train_records, dev_records, holdout_records = split_records(
        manifest["records"], config.dev_group
    )
    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
        dev_records = dev_records[: max(1, max_train_samples // 2)]
        holdout_records = holdout_records[: max(1, max_train_samples // 2)]
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable")

    datasets = {
        "train": CoarseInkDataset(
            train_records, root=dataset_root, augment=True, seed=config.seed
        ),
        "dev": CoarseInkDataset(
            dev_records, root=dataset_root, augment=False, seed=config.seed
        ),
        "holdout": CoarseInkDataset(
            holdout_records, root=dataset_root, augment=False, seed=config.seed
        ),
    }
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    evaluation_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name in ("dev", "holdout")
    }
    model = CoarseInkNet(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )
    positive_weight = _positive_weight(train_records, dataset_root)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best-model.pt"
    history: list[dict[str, Any]] = []
    best_ap = -math.inf
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            supervision = batch["supervision"].to(device, non_blocking=True)
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
        dev_result = evaluate_orientation_pair(
            model, evaluation_loaders["dev"], device
        )
        epoch_record = {
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "dev": dev_result,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch + 1} loss={epoch_record['training_loss']:.6f} "
            f"dev_ap={dev_result['forward']['average_precision']:.6f} "
            f"dev_auc={dev_result['forward']['auroc']:.6f} "
            f"orient_r={dev_result['orientation']['pearson']:.6f}",
            flush=True,
        )
        current_ap = float(dev_result["forward"]["average_precision"])
        if current_ap > best_ap + 1e-5:
            best_ap = current_ap
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_state_dict": model.state_dict(),
                    "base_channels": config.base_channels,
                    "epoch": epoch + 1,
                    "dev_average_precision": best_ap,
                    "dataset_manifest_sha256": sha256_file(manifest_path),
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dev_final = evaluate_orientation_pair(model, evaluation_loaders["dev"], device)
    # The held-out group is evaluated exactly once, after model selection.
    holdout_final = evaluate_orientation_pair(
        model, evaluation_loaders["holdout"], device
    )
    gates = {
        "holdout_forward_ap": holdout_final["forward"]["average_precision"]
        >= config.minimum_holdout_ap,
        "holdout_forward_auroc": holdout_final["forward"]["auroc"]
        >= config.minimum_holdout_auroc,
        "holdout_forward_f1": holdout_final["forward"]["f1"]
        >= config.minimum_holdout_f1,
        "orientation_pearson": holdout_final["orientation"]["pearson"]
        >= config.minimum_orientation_pearson,
        "orientation_top5_jaccard": holdout_final["orientation"]["top5_jaccard"]
        >= config.minimum_orientation_top5_jaccard,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass_eligible_inference_gate" if all(gates.values()) else "fail_closed",
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "split_groups": {
            "train": sorted({r["group"] for r in train_records}),
            "dev": sorted({r["group"] for r in dev_records}),
            "holdout": sorted({r["group"] for r in holdout_records}),
        },
        "split_counts": {
            "train": len(train_records),
            "dev": len(dev_records),
            "holdout": len(holdout_records),
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "positive_weight": positive_weight,
        "selected_epoch": checkpoint["epoch"],
        "history": history,
        "dev_final": dev_final,
        "holdout_final": holdout_final,
        "gates": gates,
        "elapsed_seconds": time.time() - started,
        "checkpoint_path": str(best_checkpoint),
        "checkpoint_sha256": sha256_file(best_checkpoint),
    }
    report_path = output_dir / "training-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=12)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        patience=args.patience,
    )
    report = train_model(
        args.manifest,
        args.dataset_root,
        args.output,
        config,
        device_name=args.device,
        max_train_samples=args.max_train_samples,
    )
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"]}, indent=2))
    return 0 if report["verdict"] == "pass_eligible_inference_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
