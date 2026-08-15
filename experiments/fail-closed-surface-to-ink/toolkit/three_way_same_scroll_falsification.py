#!/usr/bin/env python3
"""Run the frozen three-way same-scroll coarse-ink falsification protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_coarse_ink_detector import (
    CoarseInkDataset,
    CoarseInkNet,
    _positive_weight,
    calculate_metrics,
    infer_loader,
    masked_bce_dice_loss,
    set_deterministic,
    top_fraction_jaccard,
)


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported protocol schema")
    if protocol.get("experiment") != "same_scroll_three_way_falsification":
        raise ValueError("unexpected experiment")
    if protocol["sealed_cross_scroll_group"] != "841":
        raise ValueError("PHerc841 must remain the sealed cross-scroll group")
    if protocol["training"]["augmentation"] is not False:
        raise ValueError("protocol requires augmentation=false")
    if protocol["unseal_gate"]["distant_c_loaded_only_after_collective_gate"] is not True:
        raise ValueError("protocol must keep C sealed until the collective gate")
    if protocol["prohibitions"] != {
        "hyperparameter_changes_after_local_b": True,
        "pherc841_access": True,
        "eligible_scroll_inference": True,
        "runpod_or_paid_compute": True,
    }:
        raise ValueError("protocol prohibitions changed")
    return protocol


def partition_records(
    records: Iterable[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    group = protocol["group"]
    group_records = [
        record
        for record in records
        if record["group"] == group and record["split"] == "train"
    ]
    if not group_records:
        raise ValueError(f"no records for group {group}")
    if any(record["group"] == protocol["sealed_cross_scroll_group"] for record in group_records):
        raise ValueError("sealed cross-scroll records entered the experiment")
    partitions = protocol["partitions"]
    a_max = int(partitions["train_a"]["maximum_index_inclusive"])
    ab_min = int(partitions["buffer_ab"]["minimum_index_inclusive"])
    ab_max = int(partitions["buffer_ab"]["maximum_index_inclusive"])
    b_min = int(partitions["local_b"]["minimum_index_inclusive"])
    b_max = int(partitions["local_b"]["maximum_index_inclusive"])
    bc_min = int(partitions["buffer_bc"]["minimum_index_inclusive"])
    bc_max = int(partitions["buffer_bc"]["maximum_index_inclusive"])
    c_min = int(partitions["distant_c"]["minimum_index_inclusive"])
    if not (a_max < ab_min <= ab_max < b_min <= b_max < bc_min <= bc_max < c_min):
        raise ValueError("partition intervals are not strictly ordered")

    def x(record: dict[str, Any]) -> int:
        return int(record["x_index"])

    result = {
        "train_a": [record for record in group_records if x(record) <= a_max],
        "buffer_ab": [record for record in group_records if ab_min <= x(record) <= ab_max],
        "local_b": [record for record in group_records if b_min <= x(record) <= b_max],
        "buffer_bc": [record for record in group_records if bc_min <= x(record) <= bc_max],
        "distant_c": [record for record in group_records if x(record) >= c_min],
    }
    if any(not result[name] for name in ("train_a", "local_b", "distant_c")):
        raise ValueError("A, B, and C must all contain records")
    assigned_ids = {
        record["sample_id"]
        for name, items in result.items()
        for record in items
        if name != "buffer_bc" or items
    }
    if assigned_ids != {record["sample_id"] for record in group_records}:
        raise ValueError("partition rules do not account for every group record")
    region_ids = [
        {record["sample_id"] for record in result[name]}
        for name in ("train_a", "local_b", "distant_c")
    ]
    if any(region_ids[i] & region_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("A/B/C sample leakage")

    max_a_x1 = max(int(record["x1"]) for record in result["train_a"])
    min_b_x0 = min(int(record["x0"]) for record in result["local_b"])
    max_b_x1 = max(int(record["x1"]) for record in result["local_b"])
    min_c_x0 = min(int(record["x0"]) for record in result["distant_c"])
    voxel_sizes = {float(record["coarse_voxel_um"]) for record in group_records}
    if len(voxel_sizes) != 1:
        raise ValueError("inconsistent voxel sizes")
    voxel_um = voxel_sizes.pop()
    gap_ab = min_b_x0 - max_a_x1
    gap_bc = min_c_x0 - max_b_x1
    if gap_ab <= 0 or gap_bc <= gap_ab:
        raise ValueError("spatial buffers are not positive and increasingly distant")
    evidence = {
        "group": group,
        "axis": "x",
        "coarse_voxel_um": voxel_um,
        "counts": {name: len(items) for name, items in result.items()},
        "a_to_b_gap_pixels": gap_ab,
        "a_to_b_gap_mm": gap_ab * voxel_um / 1000.0,
        "b_to_c_gap_pixels": gap_bc,
        "b_to_c_gap_mm": gap_bc * voxel_um / 1000.0,
        "a_to_c_gap_pixels": min_c_x0 - max_a_x1,
        "a_to_c_gap_mm": (min_c_x0 - max_a_x1) * voxel_um / 1000.0,
        "sample_ids": {
            name: sorted(record["sample_id"] for record in items)
            for name, items in result.items()
        },
        "pherc841_loaded": False,
    }
    return result, evidence


def calibration_metrics(target: np.ndarray, score: np.ndarray, bins: int = 15) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    if target.shape != score.shape or target.size == 0:
        raise ValueError("aligned nonempty target and score are required")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, score, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    rows = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(np.count_nonzero(selected))
        if count:
            confidence = float(np.mean(score[selected]))
            observed = float(np.mean(target[selected]))
            ece += count / target.size * abs(confidence - observed)
        else:
            confidence = None
            observed = None
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": confidence,
                "observed_fraction": observed,
            }
        )
    return {
        "brier_score": float(np.mean((score - target) ** 2)),
        "ece": float(ece),
        "bin_count": bins,
        "bins": rows,
    }


def precision_recall_curve_201(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.uint8).ravel() > 0
    score = np.asarray(score, dtype=np.float64).ravel()
    points = []
    for threshold in np.linspace(0.0, 1.0, 201):
        prediction = score >= threshold
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        points.append(
            {
                "threshold": float(threshold),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
            }
        )
    return {"point_count": len(points), "points": points}


@torch.no_grad()
def evaluate_region(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    target, forward, forward_ids = infer_loader(model, loader, device, reverse_depth=False)
    reverse_target, reverse, reverse_ids = infer_loader(model, loader, device, reverse_depth=True)
    if forward_ids != reverse_ids or not np.array_equal(target, reverse_target):
        raise ValueError("forward/reverse evaluation mismatch")
    pearson = 0.0 if np.std(forward) == 0 or np.std(reverse) == 0 else float(np.corrcoef(forward, reverse)[0, 1])
    return {
        "forward": calculate_metrics(target, forward),
        "reverse": calculate_metrics(target, reverse),
        "orientation": {
            "pearson": pearson,
            "mean_absolute_difference": float(np.mean(np.abs(forward - reverse))),
            "top5_jaccard": top_fraction_jaccard(forward, reverse, 0.05),
            "top1_jaccard": top_fraction_jaccard(forward, reverse, 0.01),
        },
        "calibration": calibration_metrics(target, forward),
        "precision_recall_curve": precision_recall_curve_201(target, forward),
        "sample_count": len(forward_ids),
        "sample_ids": forward_ids,
        "supervised_pixel_count": int(target.size),
    }


def make_loader(
    records: list[dict[str, Any]], dataset_root: Path, seed: int, batch_size: int
) -> DataLoader:
    return DataLoader(
        CoarseInkDataset(records, root=dataset_root, augment=False, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def train_one_seed(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    dataset_root: Path,
    protocol: dict[str, Any],
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], Path]:
    set_deterministic(seed)
    training = protocol["training"]
    batch_size = int(training["batch_size"])
    dataset_a = CoarseInkDataset(records_a, root=dataset_root, augment=False, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader_a_train = DataLoader(
        dataset_a,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    loader_a_eval = make_loader(records_a, dataset_root, seed, batch_size)
    loader_b_eval = make_loader(records_b, dataset_root, seed, batch_size)
    model = CoarseInkNet(int(protocol["model"]["base_channels"])).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(protocol["model"]["expected_parameter_count"]):
        raise ValueError("model parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positive_weight = _positive_weight(records_a, dataset_root)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    started = time.time()
    for epoch in range(int(training["epochs"])):
        model.train()
        losses = []
        for batch in loader_a_train:
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
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "training_loss": mean_loss})
        print(f"seed={seed} epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
    a_result = evaluate_region(model, loader_a_eval, device)
    b_result = evaluate_region(model, loader_b_eval, device)
    gate = protocol["unseal_gate"]
    gates = {
        "train_a_auroc": a_result["forward"]["auroc"]
        >= float(gate["per_seed_train_a_auroc_minimum"]),
        "local_b_auroc": b_result["forward"]["auroc"]
        >= float(gate["per_seed_local_b_auroc_minimum"]),
    }
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=False, exist_ok=False)
    checkpoint_path = seed_dir / "fixed-model.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "base_channels": int(protocol["model"]["base_channels"]),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "history": history,
        "positive_weight": positive_weight,
        "train_a": a_result,
        "local_b": b_result,
        "unseal_gates": gates,
        "passes_collective_unit_gate": all(gates.values()),
        "distant_c_evaluated": False,
        "elapsed_seconds": time.time() - started,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = seed_dir / "a-b-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report, checkpoint_path


def summarize_seed_variance(seed_reports: list[dict[str, Any]], region: str) -> dict[str, Any]:
    fields = {
        "average_precision": [r[region]["forward"]["average_precision"] for r in seed_reports],
        "prevalence": [r[region]["forward"]["prevalence"] for r in seed_reports],
        "auroc": [r[region]["forward"]["auroc"] for r in seed_reports],
        "f1": [r[region]["forward"]["f1"] for r in seed_reports],
        "brier_score": [r[region]["calibration"]["brier_score"] for r in seed_reports],
        "ece": [r[region]["calibration"]["ece"] for r in seed_reports],
        "orientation_pearson": [r[region]["orientation"]["pearson"] for r in seed_reports],
        "orientation_top5_jaccard": [r[region]["orientation"]["top5_jaccard"] for r in seed_reports],
    }
    return {
        key: {
            "values": values,
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        for key, values in fields.items()
    }


def run(
    manifest_path: Path,
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    device_name: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    protocol = load_protocol(protocol_path)
    manifest = json.loads(manifest_path.read_text())
    partitions, split_evidence = partition_records(manifest["records"], protocol)
    output_dir.mkdir(parents=True)
    protocol_copy = output_dir / "locked-protocol.json"
    shutil.copyfile(protocol_path, protocol_copy)
    lock = {
        "schema_version": SCHEMA_VERSION,
        "locked_before_training": True,
        "protocol_sha256": sha256_file(protocol_copy),
        "manifest_sha256": sha256_file(manifest_path),
        "split_evidence": split_evidence,
        "c_npz_payload_loaded_before_gate": False,
        "c_predictions_inspected_before_gate": False,
        "metadata_caveat": "C sample metadata exists in the shared source manifest; C NPZ voxels and model predictions are gate-sealed.",
    }
    lock_path = output_dir / "protocol-lock-evidence.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    device = torch.device(device_name)
    seed_reports = []
    checkpoints = []
    for seed in protocol["training"]["seeds"]:
        report, checkpoint = train_one_seed(
            partitions["train_a"],
            partitions["local_b"],
            dataset_root,
            protocol,
            int(seed),
            output_dir,
            device,
        )
        seed_reports.append(report)
        checkpoints.append(checkpoint)
    passing = sum(bool(report["passes_collective_unit_gate"]) for report in seed_reports)
    required = int(protocol["unseal_gate"]["minimum_passing_seeds"])
    c_unsealed = passing >= required
    if c_unsealed:
        for report, checkpoint_path in zip(seed_reports, checkpoints):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model = CoarseInkNet(int(protocol["model"]["base_channels"])).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            c_loader = make_loader(
                partitions["distant_c"],
                dataset_root,
                int(report["seed"]),
                int(protocol["training"]["batch_size"]),
            )
            c_result = evaluate_region(model, c_loader, device)
            support = protocol["distant_support_gate"]
            c_gates = {
                "auroc": c_result["forward"]["auroc"]
                >= float(support["per_seed_auroc_minimum"]),
                "ap_above_prevalence": (
                    c_result["forward"]["average_precision"]
                    - c_result["forward"]["prevalence"]
                    >= float(support["per_seed_ap_above_prevalence_minimum"])
                ),
            }
            report["distant_c"] = c_result
            report["distant_c_gates"] = c_gates
            report["distant_c_evaluated"] = True
            seed_path = output_dir / f"seed-{report['seed']}" / "a-b-c-report.json"
            seed_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    distant_passing = (
        sum(all(report["distant_c_gates"].values()) for report in seed_reports)
        if c_unsealed
        else 0
    )
    if not c_unsealed:
        train_passes = sum(report["unseal_gates"]["train_a_auroc"] for report in seed_reports)
        verdict = (
            "stop_local_transfer_failed"
            if train_passes >= required
            else "stop_insufficient_training_fit"
        )
    elif distant_passing >= int(protocol["distant_support_gate"]["minimum_passing_seeds"]):
        verdict = "same_scroll_spatial_signal_supported"
    else:
        verdict = "stop_coarse_supervised_ssl_branch"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "protocol_sha256": sha256_file(protocol_copy),
        "protocol_lock_evidence_sha256": sha256_file(lock_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_evidence": split_evidence,
        "seed_count": len(seed_reports),
        "passing_a_b_seed_count": passing,
        "required_a_b_seed_count": required,
        "distant_c_unsealed": c_unsealed,
        "distant_c_seed_count": len(seed_reports) if c_unsealed else 0,
        "distant_c_passing_seed_count": distant_passing,
        "variance": {
            "train_a": summarize_seed_variance(seed_reports, "train_a"),
            "local_b": summarize_seed_variance(seed_reports, "local_b"),
            **(
                {"distant_c": summarize_seed_variance(seed_reports, "distant_c")}
                if c_unsealed
                else {}
            ),
        },
        "seed_reports": [
            {
                "seed": report["seed"],
                "a_b_report": f"seed-{report['seed']}/a-b-report.json",
                "a_b_report_sha256": sha256_file(
                    output_dir / f"seed-{report['seed']}" / "a-b-report.json"
                ),
                "a_b_c_report": (
                    f"seed-{report['seed']}/a-b-c-report.json" if c_unsealed else None
                ),
                "a_b_c_report_sha256": (
                    sha256_file(output_dir / f"seed-{report['seed']}" / "a-b-c-report.json")
                    if c_unsealed
                    else None
                ),
                "checkpoint_sha256": report["checkpoint_sha256"],
            }
            for report in seed_reports
        ],
        "pherc841_loaded": False,
        "eligible_scroll_inference_started": False,
        "paid_compute_usd": 0.0,
    }
    summary_path = output_dir / "three-way-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        args.manifest,
        args.dataset_root,
        args.protocol,
        args.output,
        device_name=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["verdict"] == "same_scroll_spatial_signal_supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
