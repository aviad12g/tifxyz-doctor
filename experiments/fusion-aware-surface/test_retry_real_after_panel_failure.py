#!/usr/bin/env python3
"""Tests for the mixed real-v5/synthetic-v4 result-blind retry controller."""

from __future__ import annotations

import json
from pathlib import Path

import retry_real_after_panel_failure as retry


def test_v4_receipt_requires_exact_pair(tmp_path: Path) -> None:
    path = tmp_path / "v4.json"
    path.write_text(
        json.dumps(
            {
                "accepted": [
                    {
                        "mode": "real",
                        "kernel_id": retry.REAL_KERNEL,
                        "kernel_version": 4,
                    },
                    {
                        "mode": "synthetic",
                        "kernel_id": retry.SYNTHETIC_KERNEL,
                        "kernel_version": 4,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert retry.load_v4_receipt(path)["accepted"][1]["mode"] == "synthetic"


def test_retry_selects_real_only_and_retains_synthetic() -> None:
    index = {
        "public_execution_plan": {"commit": retry.PUBLIC_PLAN_COMMIT},
        "public_cache_delivery": {"commit": retry.PUBLIC_DELIVERY_COMMIT},
    }
    packages = [
        {"mode": "real", "kaggle_kernel_id": retry.REAL_KERNEL},
        {"mode": "synthetic", "kaggle_kernel_id": retry.SYNTHETIC_KERNEL},
    ]
    assert retry.validate_inputs(index, packages, {}) is packages[0]
