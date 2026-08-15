from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "train_coarse_ink_detector.py"
SPEC = importlib.util.spec_from_file_location("train_coarse_ink_detector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_volume_is_finite_and_bounded() -> None:
    volume = np.arange(17 * 8 * 8, dtype=np.float32).reshape(17, 8, 8)
    normalized = MODULE.normalize_volume(volume)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert 0 <= normalized.min() <= normalized.max() <= 1


def test_model_output_shape_and_parameter_budget() -> None:
    model = MODULE.CoarseInkNet(base_channels=12)
    output = model(torch.zeros(2, 1, 17, 32, 32))
    assert output.shape == (2, 1, 32, 32)
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_000_000


def test_masked_loss_ignores_unsupervised_pixels() -> None:
    logits = torch.zeros(1, 1, 2, 2)
    target = torch.zeros_like(logits)
    supervision = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    first = MODULE.masked_bce_dice_loss(
        logits, target, supervision, positive_weight=2.0
    )
    changed = target.clone()
    changed[0, 0, 1, 1] = 1.0
    second = MODULE.masked_bce_dice_loss(
        logits, changed, supervision, positive_weight=2.0
    )
    assert torch.equal(first, second)


def test_metrics_perfect_ranking() -> None:
    target = np.array([0, 0, 1, 1], dtype=np.uint8)
    score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    metrics = MODULE.calculate_metrics(target, score)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_auroc_ties_are_rank_averaged() -> None:
    target = np.array([0, 1, 0, 1], dtype=np.uint8)
    score = np.ones(4, dtype=np.float32)
    assert MODULE.binary_auroc(target, score) == pytest.approx(0.5)


def test_top_fraction_jaccard_identity() -> None:
    score = np.arange(100, dtype=np.float32)
    assert MODULE.top_fraction_jaccard(score, score, 0.05) == pytest.approx(1.0)


def test_group_split_has_no_leakage() -> None:
    records = [
        {"split": "train", "group": "a"},
        {"split": "train", "group": "dev"},
        {"split": "holdout", "group": "hold"},
    ]
    train, dev, holdout = MODULE.split_records(records, "dev")
    assert [record["group"] for record in train] == ["a"]
    assert [record["group"] for record in dev] == ["dev"]
    assert [record["group"] for record in holdout] == ["hold"]
