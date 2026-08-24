#!/usr/bin/env python3
"""Fuse two calibrated ink models across depth and orientation hypotheses.

Each orientation must provide low, middle, and high depth windows.  A stable
pixel requires both models in at least two *adjacent* windows (low+middle or
middle+high); low+high without the middle window is not sufficient.  Forward
and reverse are co-equal orientation hypotheses.  They are saved separately,
and overlap between them is reported only as a diagnostic--never as a veto.

Input maps use the canonical filename form
``{segment}-{model}-{depth}-{orientation}.probability.tif``, where ``depth``
is ``low``, ``mid``, or ``high`` and ``orientation`` is ``forward`` or
``reverse``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


DEPTHS = ("low", "mid", "high")
ORIENTATIONS = ("forward", "reverse")


def load(path: Path) -> np.ndarray:
    return tifffile.imread(path).astype(np.float32)


def score_pair(a: np.ndarray, b: np.ndarray, ta: float, tb: float, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (a >= ta) & (b >= tb) & valid
    score = np.zeros_like(a, dtype=np.float32)
    score[mask] = np.sqrt(a[mask] * b[mask])
    return score, mask


def safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa, bb = a[mask], b[mask]
    if aa.size < 2 or aa.std() == 0 or bb.std() == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    ac, bc = int(a.sum()), int(b.sum())
    return {
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else 0.0,
        "dice": 2 * inter / (ac + bc) if ac + bc else 0.0,
    }


def save_map(path: Path, array: np.ndarray) -> None:
    tifffile.imwrite(path.with_suffix(".probability.tif"), array.astype(np.float32))
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(
        path.with_suffix(".probability.png")
    )


def save_overlay(path: Path, array: np.ndarray, central: np.ndarray, valid: np.ndarray, threshold: float) -> None:
    values = central[valid]
    lo, hi = np.percentile(values, (1, 99)) if values.size else (0, 255)
    base = np.clip((central.astype(np.float32) - lo) / max(float(hi - lo), 1), 0, 1)
    rgb = np.repeat(base[..., None], 3, axis=2)
    alpha = np.clip((array - threshold) / max(1 - threshold, 1e-6), 0, 1)[..., None]
    cyan = np.zeros_like(rgb)
    cyan[..., 1:] = 1
    out = rgb * (1 - 0.82 * alpha) + cyan * (0.82 * alpha)
    out[~valid] = 0
    Image.fromarray(np.rint(out * 255).astype(np.uint8)).save(path.with_suffix(".overlay.png"))


def stable_depth_maps(
    pair_scores: dict[str, np.ndarray],
    pair_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return adjacent-2 and all-3 stable scores, masks, and band counts."""

    count = sum(pair_masks[depth].astype(np.uint8) for depth in DEPTHS)
    low_mid = pair_masks["low"] & pair_masks["mid"]
    mid_high = pair_masks["mid"] & pair_masks["high"]
    stable2_mask = low_mid | mid_high
    stable3_mask = low_mid & pair_masks["high"]
    score_sum = sum(pair_scores[depth] for depth in DEPTHS)
    stable2 = np.divide(
        score_sum,
        count,
        out=np.zeros_like(score_sum, dtype=np.float32),
        where=stable2_mask,
    ).astype(np.float32)
    stable3 = np.where(stable3_mask, score_sum / 3.0, 0).astype(np.float32)
    return stable2, stable3, stable2_mask, stable3_mask, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map_dir", type=Path)
    parser.add_argument("segment", help="Output stem used by inference, for example seg1")
    parser.add_argument("central_layer", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold-a", type=float, default=0.49)
    parser.add_argument("--threshold-b", type=float, default=0.43)
    parser.add_argument("--model-a", default="gp")
    parser.add_argument("--model-b", default="large")
    args = parser.parse_args()

    central = tifffile.imread(args.central_layer)
    valid = central > 0
    shape = central.shape
    pair_threshold = float(np.sqrt(args.threshold_a * args.threshold_b))
    report: dict[str, object] = {
        "schema_version": 2,
        "segment": args.segment,
        "shape": list(shape),
        "valid_pixels": int(valid.sum()),
        "threshold_a": args.threshold_a,
        "threshold_b": args.threshold_b,
        "pair_threshold": pair_threshold,
        "input_filename_template": "{segment}-{model}-{depth}-{orientation}.probability.tif",
        "orientation_policy": (
            "forward and reverse are co-equal; each must pass the same three-band "
            "adjacent-depth test; cross-orientation overlap is diagnostic only"
        ),
        "orientations": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stable_scores_by_orientation: dict[str, dict[str, np.ndarray]] = {}
    stable_masks_by_orientation: dict[str, dict[str, np.ndarray]] = {}
    raw_by_orientation: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for orientation in ORIENTATIONS:
        pair_scores: dict[str, np.ndarray] = {}
        pair_masks: dict[str, np.ndarray] = {}
        raw_a: dict[str, np.ndarray] = {}
        raw_b: dict[str, np.ndarray] = {}
        band_report: dict[str, object] = {}

        for depth in DEPTHS:
            band = f"{depth}-{orientation}"
            path_a = args.map_dir / f"{args.segment}-{args.model_a}-{band}.probability.tif"
            path_b = args.map_dir / f"{args.segment}-{args.model_b}-{band}.probability.tif"
            a, b = load(path_a), load(path_b)
            if a.shape != shape or b.shape != shape:
                raise ValueError(f"Shape mismatch in {band}: {a.shape}, {b.shape}, {shape}")
            score, mask = score_pair(a, b, args.threshold_a, args.threshold_b, valid)
            raw_a[depth], raw_b[depth] = a, b
            pair_scores[depth], pair_masks[depth] = score, mask
            pa, pb = (a >= args.threshold_a) & valid, (b >= args.threshold_b) & valid
            band_report[depth] = {
                "map_a": str(path_a),
                "map_b": str(path_b),
                "a_positive": int(pa.sum()),
                "b_positive": int(pb.sum()),
                "pair_positive": int(mask.sum()),
                "model_overlap": overlap(pa, pb),
                "raw_probability_correlation": safe_corr(a, b, valid),
            }
            prefix = args.output_dir / f"{args.segment}-pair-{band}"
            save_map(prefix, score)
            save_overlay(prefix, score, central, valid, pair_threshold)

        stable2, stable3, stable2_mask, stable3_mask, count = stable_depth_maps(
            pair_scores, pair_masks
        )
        for name, array in (("stable2", stable2), ("stable3", stable3)):
            prefix = args.output_dir / f"{args.segment}-{orientation}-{name}"
            save_map(prefix, array)
            save_overlay(prefix, array, central, valid, pair_threshold)

        report["orientations"][orientation] = {
            "role": "co_equal_orientation_hypothesis",
            "bands": band_report,
            "depth_stability": {
                "stable2_definition": "(low AND mid) OR (mid AND high)",
                "stable3_definition": "low AND mid AND high",
                "stable2_adjacent_pixels": int(stable2_mask.sum()),
                "stable3_pixels": int(stable3_mask.sum()),
                "pixels_in_0_bands": int((count == 0).sum()),
                "pixels_in_1_band": int((count == 1).sum()),
                "pixels_in_2_bands": int((count == 2).sum()),
                "pixels_in_3_bands": int((count == 3).sum()),
                "low_mid": overlap(pair_masks["low"], pair_masks["mid"]),
                "mid_high": overlap(pair_masks["mid"], pair_masks["high"]),
                "low_high_diagnostic_only": overlap(pair_masks["low"], pair_masks["high"]),
            },
        }
        stable_scores_by_orientation[orientation] = {"stable2": stable2, "stable3": stable3}
        stable_masks_by_orientation[orientation] = {
            "stable2": stable2_mask,
            "stable3": stable3_mask,
        }
        raw_by_orientation[orientation] = {"a": raw_a, "b": raw_b}

    report["cross_orientation_diagnostic"] = {
        "used_as_veto": False,
        "policy": "overlap and correlation are diagnostic only; neither orientation suppresses the other",
        "stable2_overlap": overlap(
            stable_masks_by_orientation["forward"]["stable2"],
            stable_masks_by_orientation["reverse"]["stable2"],
        ),
        "stable3_overlap": overlap(
            stable_masks_by_orientation["forward"]["stable3"],
            stable_masks_by_orientation["reverse"]["stable3"],
        ),
        "stable2_probability_correlation": safe_corr(
            stable_scores_by_orientation["forward"]["stable2"],
            stable_scores_by_orientation["reverse"]["stable2"],
            valid,
        ),
        "stable3_probability_correlation": safe_corr(
            stable_scores_by_orientation["forward"]["stable3"],
            stable_scores_by_orientation["reverse"]["stable3"],
            valid,
        ),
        "middle_model_a_correlation": safe_corr(
            raw_by_orientation["forward"]["a"]["mid"],
            raw_by_orientation["reverse"]["a"]["mid"],
            valid,
        ),
        "middle_model_b_correlation": safe_corr(
            raw_by_orientation["forward"]["b"]["mid"],
            raw_by_orientation["reverse"]["b"]["mid"],
            valid,
        ),
    }
    manifest = args.output_dir / f"{args.segment}-depth-consensus.manifest.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
