from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from audit_pherc0800_published_segments import selected_run_metrics
from salvage_calibrated_m7_component import calibrated_vertex_mask, largest_quad_mask


def test_calibrated_mask_allows_separated_competitor_but_requires_centered_run() -> None:
    frames = np.zeros((31, 4, 4), dtype=np.uint8)
    frames[14:18] = 255  # selected run offsets -1..+2, midpoint +0.5
    frames[25:27] = 255  # separated competitor offsets +10..+11
    frames[14:18, 0, 0] = 0
    frames[20:23, 0, 0] = 255  # only off-center run at +5..+7
    eligible = np.ones((4, 4), dtype=bool)
    metrics = selected_run_metrics(frames, eligible)
    mask = calibrated_vertex_mask(metrics, eligible, maximum_center_offset=2.0)
    assert int(np.count_nonzero(mask)) == 15
    assert not mask[0, 0]
    assert mask[1, 1]
    assert metrics["run_count"][1, 1] == 2


def test_largest_quad_mask_cannot_connect_through_isolated_vertex() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:3, 0:3] = True
    mask[3, 3] = True  # diagonal point contact only
    mask[3:6, 4:8] = True
    area = np.ones((5, 7), dtype=np.float64)
    selected, record = largest_quad_mask(mask, area)
    assert record["components"]["component_count"] == 2
    assert int(np.count_nonzero(selected[:-1, :-1] & selected[:-1, 1:] & selected[1:, :-1] & selected[1:, 1:])) == 6
    assert not selected[3, 3]
