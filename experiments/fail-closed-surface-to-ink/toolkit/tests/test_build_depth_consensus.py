from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import tifffile


SCRIPT = Path(__file__).parents[1] / "build_depth_consensus.py"
SPEC = importlib.util.spec_from_file_location("build_depth_consensus", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_symmetric_orientations_require_adjacent_depth_support(
    tmp_path: Path, monkeypatch
) -> None:
    map_dir = tmp_path / "maps"
    output_dir = tmp_path / "out"
    map_dir.mkdir()
    segment = "candidate"
    central = np.full((5, 6), 160, dtype=np.uint8)
    central_path = tmp_path / "15.tif"
    tifffile.imwrite(central_path, central)

    masks = {
        "low-forward": {(0, 0), (0, 1), (0, 2)},
        "mid-forward": {(0, 0), (0, 2)},
        "high-forward": {(0, 1), (0, 2)},
        "low-reverse": {(0, 2), (1, 0)},
        "mid-reverse": {(0, 2), (1, 0), (1, 1)},
        "high-reverse": {(0, 2), (1, 1)},
    }
    for band, positives in masks.items():
        for model in ("gp", "large"):
            probability = np.full(central.shape, 0.1, dtype=np.float32)
            for y, x in positives:
                probability[y, x] = 0.8 if model == "gp" else 0.9
            tifffile.imwrite(
                map_dir / f"{segment}-{model}-{band}.probability.tif",
                probability,
            )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            str(map_dir),
            segment,
            str(central_path),
            str(output_dir),
            "--threshold-a",
            "0.5",
            "--threshold-b",
            "0.5",
        ],
    )
    MODULE.main()

    forward2 = tifffile.imread(
        output_dir / f"{segment}-forward-stable2.probability.tif"
    )
    forward3 = tifffile.imread(
        output_dir / f"{segment}-forward-stable3.probability.tif"
    )
    reverse2 = tifffile.imread(
        output_dir / f"{segment}-reverse-stable2.probability.tif"
    )
    reverse3 = tifffile.imread(
        output_dir / f"{segment}-reverse-stable3.probability.tif"
    )

    # Forward (0,0) is low+mid and passes.  Forward (0,1) is low+high
    # without the adjacent middle band and must not pass.
    assert forward2[0, 0] > 0.5
    assert forward2[0, 1] == 0
    assert forward3[0, 2] > 0.5

    # Reverse is evaluated with the same rule, and a pixel shared by both
    # orientations remains present in both outputs (no orientation veto).
    assert reverse2[1, 0] > 0.5
    assert reverse2[1, 1] > 0.5
    assert reverse3[0, 2] > 0.5
    assert forward3[0, 2] > 0.5

    report = json.loads(
        (output_dir / f"{segment}-depth-consensus.manifest.json").read_text()
    )
    assert report["schema_version"] == 2
    assert set(report["orientations"]) == {"forward", "reverse"}
    assert (
        report["orientations"]["forward"]["depth_stability"][
            "stable2_definition"
        ]
        == "(low AND mid) OR (mid AND high)"
    )
    assert report["cross_orientation_diagnostic"]["used_as_veto"] is False
    assert (
        report["cross_orientation_diagnostic"]["stable3_overlap"]["intersection"]
        == 1
    )


def test_stable_score_averages_only_supporting_bands() -> None:
    scores = {
        "low": np.array([[0.6]], dtype=np.float32),
        "mid": np.array([[0.8]], dtype=np.float32),
        "high": np.array([[0.0]], dtype=np.float32),
    }
    masks = {
        "low": np.array([[True]]),
        "mid": np.array([[True]]),
        "high": np.array([[False]]),
    }
    stable2, stable3, stable2_mask, stable3_mask, count = MODULE.stable_depth_maps(
        scores, masks
    )
    assert count[0, 0] == 2
    assert stable2_mask[0, 0]
    assert not stable3_mask[0, 0]
    assert np.isclose(stable2[0, 0], 0.7)
    assert stable3[0, 0] == 0
