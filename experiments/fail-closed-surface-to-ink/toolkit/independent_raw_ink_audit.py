#!/usr/bin/env python3
"""Model-independent ink triage for a 93-layer surface-normal stack.

This is deliberately not a text or prize validator.  It searches for dark,
sheet-relative contrast that persists through adjacent near-surface offsets,
is stronger near the nominal sheet than in distant offsets, and is not well
explained by a single long papyrus-fiber orientation.  Surviving components
are tested for row-like topology and rendered for human review.

No learned model, model probability map, network service, or GPU is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.filters import frangi
from skimage.morphology import disk, remove_small_objects, skeletonize


STACK_DEPTH = 93
CENTER_INDEX = 46
EXACT_OFFSETS_VOXELS = tuple(range(-CENTER_INDEX, STACK_DEPTH - CENTER_INDEX))
NEAR_BLOCKS = ((32, 39), (39, 46), (46, 53), (53, 60))
NEAR_INDICES = tuple(range(32, 60))
FAR_INDICES = (4, 8, 12, 16, 20, 24, 68, 72, 76, 80, 84, 88)
REVIEW_DEPTH_INDICES = (15, 31, 39, 46, 53, 61, 77)
ROW_ANGLE_DEGREES = tuple(range(-90, 90, 6))
MAXIMUM_ROW_CONTEXT_UM = 20_000.0
DEFAULT_VOXEL_UM = 8.64
OUTPUT_FLOAT_MAPS = (
    "adjacent_depth_persistence.tif",
    "far_depth_persistence.tif",
    "near_depth_support.tif",
    "surface_specificity.tif",
    "fiber_ridge_likelihood.tif",
    "fiber_likelihood.tif",
    "independent_ink_score.tif",
)


class IndependentInkAuditError(RuntimeError):
    """Raised when the audit cannot produce trustworthy evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_exact_stack(stack_dir: Path) -> tuple[np.ndarray, dict[str, str]]:
    layers: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    shape: tuple[int, int] | None = None
    for index in range(STACK_DEPTH):
        path = stack_dir / f"{index:02d}.tif"
        if not path.is_file():
            raise IndependentInkAuditError(f"missing stack layer {index:02d}: {path}")
        layer = np.asarray(tifffile.imread(path))
        if layer.ndim != 2 or layer.dtype != np.uint8:
            raise IndependentInkAuditError(
                f"stack layer {index:02d} must be 2-D uint8, got {layer.dtype} {layer.shape}"
            )
        if shape is None:
            shape = layer.shape
        elif layer.shape != shape:
            raise IndependentInkAuditError(
                f"stack layer {index:02d} shape {layer.shape} differs from {shape}"
            )
        layers.append(layer)
        hashes[path.name] = sha256_file(path)
    return np.stack(layers, axis=0), hashes


def load_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.is_file():
        raise IndependentInkAuditError(f"missing explicit mask: {mask_path}")
    mask = np.asarray(tifffile.imread(mask_path))
    if mask.ndim != 2 or mask.shape != shape:
        raise IndependentInkAuditError(
            f"mask shape {mask.shape} differs from stack shape {shape}"
        )
    result = mask != 0
    if not np.any(result):
        raise IndependentInkAuditError("explicit mask contains no valid pixels")
    return result


def validate_exact_93_manifest(
    manifest_path: Path,
    stack_dir: Path,
    layer_hashes: Mapping[str, str],
    stack_shape: tuple[int, int, int],
    mask_path: Path,
    mask: np.ndarray,
    *,
    voxel_um: float,
) -> dict[str, Any]:
    """Fail closed on the immutable 93-layer frame/offset/hash contract."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentInkAuditError(
            f"cannot read exact input manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise IndependentInkAuditError("exact input manifest is not a JSON object")
    if manifest.get("status") != "complete":
        raise IndependentInkAuditError("exact input manifest status is not complete")
    raw = manifest.get("raw_stack")
    if not isinstance(raw, dict):
        raise IndependentInkAuditError("exact input manifest has no raw_stack object")

    expected_shape = [STACK_DEPTH, int(stack_shape[1]), int(stack_shape[2])]
    offsets_voxels = raw.get("offsets_voxels", raw.get("stored_offsets_voxels"))
    offsets_micrometers = raw.get(
        "offsets_micrometers", raw.get("stored_offsets_micrometers")
    )
    mappings = manifest.get("sign_order_and_model_window_mappings", {})
    full_equivalences = mappings.get("full_93_layer_sign_order_equivalences", {})
    negative_reversal_ok = raw.get("negative_stack_policy") == (
        "not stored; exact layer reversal 92..0"
    ) or full_equivalences.get("negative_direct") == list(range(92, -1, -1))
    checks = {
        "layer_count": raw.get("layer_count", raw.get("shape_frame_row_column", [None])[0])
        == STACK_DEPTH,
        "shape_frame_row_column": raw.get("shape_frame_row_column") == expected_shape,
        "dtype_uint8": raw.get("dtype") == "uint8",
        "directory_identity": Path(str(raw.get("directory", ""))).resolve()
        == stack_dir.resolve(),
        "offsets_voxels_exact_minus46_through_plus46": offsets_voxels
        == list(EXACT_OFFSETS_VOXELS),
        "center_frame_46_is_offset_zero": (
            isinstance(offsets_voxels, list)
            and len(offsets_voxels) == STACK_DEPTH
            and offsets_voxels[CENTER_INDEX] == 0
        ),
        "negative_stack_exact_reversal_policy": negative_reversal_ok,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise IndependentInkAuditError(
            "exact input manifest contract failed: " + ", ".join(failed)
        )

    expected_um = [float(offset) * voxel_um for offset in EXACT_OFFSETS_VOXELS]
    actual_um = offsets_micrometers
    if not (
        isinstance(actual_um, list)
        and len(actual_um) == STACK_DEPTH
        and all(float(actual) == expected for actual, expected in zip(actual_um, expected_um, strict=True))
    ):
        raise IndependentInkAuditError(
            "exact input manifest micrometer mapping is not frame[i]=(i-46)*voxel_um"
        )

    records = raw.get("layers")
    renderer_root = stack_dir.parent
    if records is None and raw.get("renderer_manifest_path") is not None:
        renderer_manifest_path = Path(str(raw["renderer_manifest_path"]))
        renderer_payload = renderer_manifest_path.read_bytes()
        if hashlib.sha256(renderer_payload).hexdigest() != raw.get(
            "renderer_manifest_sha256"
        ):
            raise IndependentInkAuditError("nested renderer manifest hash mismatch")
        renderer = json.loads(renderer_payload)
        records = renderer.get("outputs_by_normal_sign", {}).get("positive", {}).get(
            "layers"
        )
        renderer_root = renderer_manifest_path.parent
    if not isinstance(records, list) or len(records) != STACK_DEPTH:
        raise IndependentInkAuditError("exact input manifest does not contain 93 layer records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise IndependentInkAuditError(f"layer record {index} is not an object")
        path = stack_dir / f"{index:02d}.tif"
        expected = {
            "frame": index,
            "offset_voxels": EXACT_OFFSETS_VOXELS[index],
            "path": path.resolve(),
            "sha256": layer_hashes[path.name],
            "bytes": path.stat().st_size,
        }
        if record.get("frame") != expected["frame"]:
            raise IndependentInkAuditError(f"layer record {index} has the wrong frame")
        if record.get("offset_voxels", EXACT_OFFSETS_VOXELS[index]) != expected["offset_voxels"]:
            raise IndependentInkAuditError(f"layer record {index} has the wrong offset")
        recorded_path = Path(str(record.get("path", "")))
        resolved_recorded_path = (
            recorded_path.resolve()
            if recorded_path.is_absolute()
            else (renderer_root / recorded_path).resolve()
        )
        if resolved_recorded_path != expected["path"]:
            raise IndependentInkAuditError(f"layer record {index} has the wrong path")
        if record.get("sha256") != expected["sha256"]:
            raise IndependentInkAuditError(f"layer record {index} hash mismatch")
        if record.get("bytes") != expected["bytes"]:
            raise IndependentInkAuditError(f"layer record {index} byte-count mismatch")

    renderer_mask_path = stack_dir / "valid-all.tif"
    if not renderer_mask_path.is_file():
        raise IndependentInkAuditError("renderer valid-all.tif is missing")
    renderer_mask = np.asarray(tifffile.imread(renderer_mask_path))
    if renderer_mask.shape != mask.shape or not np.array_equal(renderer_mask != 0, mask):
        raise IndependentInkAuditError(
            "renderer valid-all mask differs from the explicit audited mask"
        )

    return {
        "status": "pass",
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_id": manifest.get("candidate_id"),
        "contract_checks": checks,
        "frame_count": STACK_DEPTH,
        "center_index": CENTER_INDEX,
        "center_offset_voxels": 0,
        "voxel_um": voxel_um,
        "offsets_voxels": list(EXACT_OFFSETS_VOXELS),
        "layer_record_hashes_verified": STACK_DEPTH,
        "renderer_valid_all_mask": {
            "path": str(renderer_mask_path.resolve()),
            "sha256": sha256_file(renderer_mask_path),
            "identical_to_explicit_mask": True,
        },
        "explicit_mask": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
        },
    }


def normalized_uniform_mean(
    image: np.ndarray, mask: np.ndarray, size: int
) -> np.ndarray:
    """Mask-normalized box mean; holes never enter the local baseline as zero."""

    if size <= 1 or size % 2 == 0:
        raise ValueError("local background size must be an odd integer > 1")
    weights = ndimage.uniform_filter(
        mask.astype(np.float32), size=size, mode="nearest"
    )
    numerator = ndimage.uniform_filter(
        image.astype(np.float32) * mask, size=size, mode="nearest"
    )
    return np.divide(
        numerator,
        weights,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=weights > 1.0e-5,
    )


def robust_dark_residual(
    image: np.ndarray,
    mask: np.ndarray,
    interior: np.ndarray,
    *,
    background_size: int = 41,
    denoise_sigma: float = 1.1,
    clip_z: float = 8.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Positive robust z-score for pixels darker than their sheet neighborhood."""

    numeric = image.astype(np.float32)
    smooth_weights = ndimage.gaussian_filter(
        mask.astype(np.float32), sigma=denoise_sigma, mode="nearest"
    )
    smooth_numerator = ndimage.gaussian_filter(
        numeric * mask, sigma=denoise_sigma, mode="nearest"
    )
    smooth = np.divide(
        smooth_numerator,
        smooth_weights,
        out=np.zeros_like(smooth_numerator, dtype=np.float32),
        where=smooth_weights > 1.0e-5,
    )
    background = normalized_uniform_mean(smooth, mask, background_size)
    residual = background - smooth
    sample = residual[interior]
    if sample.size == 0:
        raise IndependentInkAuditError("interior mask contains no pixels")
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    robust_sigma = max(1.4826 * mad, 0.25)
    dark = np.clip((residual - median) / robust_sigma, 0.0, clip_z).astype(
        np.float32
    )
    dark[~interior] = 0.0
    return dark, {
        "residual_median": median,
        "residual_mad": mad,
        "robust_sigma": robust_sigma,
    }


def aggregate_depth_evidence(
    near_residuals: np.ndarray,
    far_residuals: np.ndarray,
    *,
    support_threshold_z: float = 0.8,
) -> dict[str, np.ndarray]:
    """Combine near-offset persistence with a distant-offset negative control."""

    if near_residuals.ndim != 3 or near_residuals.shape[0] != len(NEAR_INDICES):
        raise ValueError("near residual stack has the wrong shape")
    if far_residuals.ndim != 3 or far_residuals.shape[0] != len(FAR_INDICES):
        raise ValueError("far residual stack has the wrong shape")
    block_medians = []
    near_start = NEAR_INDICES[0]
    for start, end in NEAR_BLOCKS:
        block_medians.append(
            np.median(
                near_residuals[start - near_start : end - near_start], axis=0
            ).astype(np.float32)
        )
    # Require strength in two adjacent seven-layer blocks.  A one-frame crack
    # or interpolation spike cannot pass this operation.
    adjacent = np.maximum.reduce(
        [np.minimum(left, right) for left, right in zip(block_medians[:-1], block_medians[1:], strict=True)]
    ).astype(np.float32)
    support = np.mean(
        near_residuals >= support_threshold_z, axis=0, dtype=np.float32
    ).astype(np.float32)
    far = np.median(far_residuals, axis=0).astype(np.float32)
    # A feature equally dark through distant normal offsets is more consistent
    # with a through-sheet fiber/crack than a surface-local deposit.
    specificity = np.divide(
        adjacent,
        adjacent + 0.75 * far + 0.5,
        out=np.zeros_like(adjacent),
        where=(adjacent + 0.75 * far + 0.5) > 0,
    ).astype(np.float32)
    raw = adjacent * np.sqrt(support) * specificity
    return {
        "adjacent_persistence": adjacent,
        "near_support": support,
        "far_persistence": far,
        "surface_specificity": specificity,
        "raw_depth_score": raw.astype(np.float32),
    }


def robust_unit_interval(
    array: np.ndarray,
    valid: np.ndarray,
    *,
    low_percentile: float = 70.0,
    high_percentile: float = 99.7,
) -> tuple[np.ndarray, dict[str, float]]:
    values = array[valid]
    if values.size == 0:
        raise IndependentInkAuditError("cannot normalize an empty valid region")
    low, high = (
        float(value)
        for value in np.percentile(values, (low_percentile, high_percentile))
    )
    high = max(high, low + 1.0e-6)
    normalized = np.clip((array.astype(np.float32) - low) / (high - low), 0, 1)
    normalized[~valid] = 0.0
    return normalized.astype(np.float32), {"low": low, "high": high}


def fiber_orientation_evidence(
    score: np.ndarray,
    valid: np.ndarray,
    *,
    tensor_sigma: float = 24.0,
) -> dict[str, np.ndarray]:
    """Estimate long-range single-orientation texture using a structure tensor."""

    smoothed = ndimage.gaussian_filter(score, sigma=1.2, mode="nearest")
    gy, gx = np.gradient(smoothed)
    jxx = ndimage.gaussian_filter(gx * gx, sigma=tensor_sigma, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, sigma=tensor_sigma, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, sigma=tensor_sigma, mode="nearest")
    trace = jxx + jyy
    discriminant = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = np.divide(
        discriminant,
        trace + 1.0e-8,
        out=np.zeros_like(trace, dtype=np.float32),
        where=trace > 1.0e-8,
    )
    energy, _ = robust_unit_interval(
        np.sqrt(np.maximum(trace, 0.0)),
        valid,
        low_percentile=50.0,
        high_percentile=99.0,
    )
    directional = np.clip(coherence * energy, 0.0, 1.0).astype(np.float32)
    # Multi-scale Hessian ridges catch thin, locally curved papyrus fibers that
    # the larger structure tensor intentionally smooths over.  This is a
    # conservative negative control: true ink strokes can also be ridge-like,
    # so the response is retained as an explicit diagnostic and never used by
    # itself to claim or reject text.
    ridge_raw = frangi(
        score.astype(np.float32),
        sigmas=(1, 2, 3, 4),
        black_ridges=False,
    ).astype(np.float32)
    ridge, _ = robust_unit_interval(
        ridge_raw,
        valid,
        low_percentile=75.0,
        high_percentile=99.5,
    )
    likelihood = np.maximum(directional, ridge).astype(np.float32)
    # Tensor orientation modulo pi, encoded in [0, pi).
    orientation = (0.5 * np.arctan2(2.0 * jxy, jxx - jyy)) % np.pi
    coherence[~valid] = 0.0
    likelihood[~valid] = 0.0
    orientation[~valid] = 0.0
    return {
        "coherence": coherence.astype(np.float32),
        "energy": energy.astype(np.float32),
        "directional_fiber_likelihood": directional,
        "ridge_fiber_likelihood": ridge,
        "fiber_likelihood": likelihood,
        "orientation_radians": orientation.astype(np.float32),
    }


def component_features(
    binary: np.ndarray,
    score: np.ndarray,
    fiber_likelihood: np.ndarray,
) -> list[dict[str, Any]]:
    cleaned = ndimage.binary_closing(binary, structure=disk(2))
    cleaned = remove_small_objects(cleaned, min_size=14, connectivity=2)
    labels, _ = ndimage.label(cleaned, structure=np.ones((3, 3), dtype=np.uint8))
    components: list[dict[str, Any]] = []
    for label_id, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        yy, xx = slices
        local = labels[yy, xx] == label_id
        area = int(np.count_nonzero(local))
        height = int(yy.stop - yy.start)
        width = int(xx.stop - xx.start)
        if area < 14 or area > 25_000 or width < 3 or height < 3:
            continue
        ys, xs = np.nonzero(local)
        points = np.column_stack((xs, ys)).astype(np.float64)
        centred = points - points.mean(axis=0, keepdims=True)
        covariance = centred.T @ centred / max(len(points), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        elongation = math.sqrt(
            (float(eigenvalues[-1]) + 1.0) / (float(eigenvalues[0]) + 1.0)
        )
        principal = eigenvectors[:, -1]
        principal_angle_degrees = math.degrees(
            math.atan2(float(principal[1]), float(principal[0]))
        ) % 180.0
        skeleton_length = int(np.count_nonzero(skeletonize(local)))
        local_score = score[yy, xx][local]
        local_fiber = fiber_likelihood[yy, xx][local]
        weights = np.maximum(local_score.astype(np.float64), 1.0e-6)
        cx = float(xx.start + np.average(xs, weights=weights))
        cy = float(yy.start + np.average(ys, weights=weights))
        long_thin = bool(elongation > 10.0 and max(width, height) > 100)
        components.append(
            {
                "label": label_id,
                "x": cx,
                "y": cy,
                "x0": int(xx.start),
                "y0": int(yy.start),
                "x1": int(xx.stop),
                "y1": int(yy.stop),
                "width": width,
                "height": height,
                "area": area,
                "skeleton_length": skeleton_length,
                "elongation": elongation,
                "principal_angle_degrees": principal_angle_degrees,
                "mean_score": float(np.mean(local_score)),
                "peak_score": float(np.max(local_score)),
                "mean_fiber_likelihood": float(np.mean(local_fiber)),
                "long_thin_fiber_proxy": long_thin,
            }
        )
    return components


def _merge_projected_positions(
    components: Sequence[dict[str, Any]],
    angle_radians: float,
    minimum_gap_pixels: float,
) -> list[dict[str, Any]]:
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    ordered = sorted(
        components,
        key=lambda item: float(item["x"]) * cosine + float(item["y"]) * sine,
    )
    merged: list[dict[str, Any]] = []
    positions: list[float] = []
    for component in ordered:
        position = float(component["x"]) * cosine + float(component["y"]) * sine
        quality = float(component["mean_score"]) * math.sqrt(float(component["area"]))
        if not merged or position - positions[-1] >= minimum_gap_pixels:
            merged.append(component)
            positions.append(position)
        else:
            previous_quality = float(merged[-1]["mean_score"]) * math.sqrt(
                float(merged[-1]["area"])
            )
            if quality > previous_quality:
                merged[-1] = component
                positions[-1] = position
    return merged


def orientation_entropy(
    orientation: np.ndarray,
    weights: np.ndarray,
    selection: np.ndarray,
    bins: int = 12,
) -> float:
    selected_weights = weights[selection].astype(np.float64)
    if selected_weights.size == 0 or float(selected_weights.sum()) <= 0:
        return 0.0
    histogram, _ = np.histogram(
        orientation[selection],
        bins=bins,
        range=(0.0, math.pi),
        weights=selected_weights,
    )
    probabilities = histogram / max(float(histogram.sum()), 1.0e-12)
    nonzero = probabilities[probabilities > 0]
    return float(-np.sum(nonzero * np.log(nonzero)) / math.log(bins))


def rank_row_hypotheses(
    components: Sequence[dict[str, Any]],
    score: np.ndarray,
    fiber_likelihood: np.ndarray,
    orientation: np.ndarray,
    interior: np.ndarray,
    *,
    voxel_um: float,
    top: int = 12,
) -> list[dict[str, Any]]:
    eligible = [
        component
        for component in components
        if not component["long_thin_fiber_proxy"]
        and float(component["mean_fiber_likelihood"]) <= 0.82
    ]
    row_half = max(24.0, 1_300.0 / voxel_um)
    minimum_gap = max(10.0, 350.0 / voxel_um)
    maximum_span = MAXIMUM_ROW_CONTEXT_UM / voxel_um
    candidates: list[dict[str, Any]] = []
    # A surface-grid axis has no guaranteed relation to the writing baseline.
    # Search the full unoriented 180-degree range rather than silently assuming
    # that text is nearly horizontal in the raster.
    for angle_degrees in ROW_ANGLE_DEGREES:
        angle = math.radians(angle_degrees)
        cosine, sine = math.cos(angle), math.sin(angle)
        projected = [
            (
                component,
                -float(component["x"]) * sine + float(component["y"]) * cosine,
            )
            for component in eligible
        ]
        for anchor, anchor_v in projected:
            band = [component for component, value in projected if abs(value - anchor_v) <= row_half]
            band.sort(
                key=lambda item: float(item["x"]) * cosine
                + float(item["y"]) * sine
            )
            if len(band) < 5:
                continue
            projected_u = [
                float(item["x"]) * cosine + float(item["y"]) * sine
                for item in band
            ]
            # Select the densest bounded 20 mm context rather than allowing a
            # fiber band to collect fragments across the entire surface.
            best_window: list[dict[str, Any]] = []
            best_window_quality = -math.inf
            right = 0
            for left in range(len(band)):
                right = max(right, left + 1)
                while right < len(band) and projected_u[right] - projected_u[left] <= maximum_span:
                    right += 1
                window = band[left:right]
                quality = len(window) + 0.01 * sum(
                    float(item["mean_score"]) * math.sqrt(float(item["area"]))
                    for item in window
                )
                if quality > best_window_quality:
                    best_window = window
                    best_window_quality = quality
            glyphs = _merge_projected_positions(best_window, angle, minimum_gap)
            if len(glyphs) < 5:
                continue
            positions = np.asarray(
                [float(item["x"]) * cosine + float(item["y"]) * sine for item in glyphs]
            )
            span_pixels = float(positions[-1] - positions[0])
            if span_pixels < 1_500.0 / voxel_um:
                continue
            gaps = np.diff(positions)
            gap_cv = float(gaps.std() / max(gaps.mean(), 1.0)) if gaps.size else 9.0
            x0 = max(0, min(int(min(item["x0"] for item in glyphs) - 40), score.shape[1] - 1))
            x1 = min(score.shape[1], max(int(max(item["x1"] for item in glyphs) + 40), x0 + 1))
            y0 = max(0, min(int(min(item["y0"] for item in glyphs) - 70), score.shape[0] - 1))
            y1 = min(score.shape[0], max(int(max(item["y1"] for item in glyphs) + 70), y0 + 1))
            selection = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            selection_score = score[y0:y1, x0:x1]
            selection[:] = (selection_score >= 0.42) & interior[y0:y1, x0:x1]
            entropy = orientation_entropy(
                orientation[y0:y1, x0:x1],
                selection_score,
                selection,
            )
            weights = selection_score[selection].astype(np.float64)
            mean_fiber = (
                float(
                    np.average(
                        fiber_likelihood[y0:y1, x0:x1][selection], weights=weights
                    )
                )
                if weights.size and float(weights.sum()) > 0
                else 1.0
            )
            mean_component_score = float(
                np.mean([float(item["mean_score"]) for item in glyphs])
            )
            fiber_dominance_ratio = mean_fiber / max(mean_component_score, 1.0e-6)
            bbox_width_mm = (x1 - x0) * voxel_um / 1000.0
            bbox_height_mm = (y1 - y0) * voxel_um / 1000.0
            bbox_area_cm2 = bbox_width_mm * bbox_height_mm / 100.0
            parallel_weights = np.asarray(
                [
                    float(item["area"]) * max(float(item["elongation"]) - 1.0, 0.0)
                    for item in glyphs
                ],
                dtype=np.float64,
            )
            component_angles = np.asarray(
                [float(item["principal_angle_degrees"]) for item in glyphs],
                dtype=np.float64,
            )
            angle_differences = np.abs(
                ((component_angles - angle_degrees + 90.0) % 180.0) - 90.0
            )
            row_parallel_fraction = (
                float(np.sum(parallel_weights[angle_differences <= 15.0]) / parallel_weights.sum())
                if float(parallel_weights.sum()) > 0
                else 0.0
            )
            quality = (
                len(glyphs)
                + 1.5 * min(span_pixels * voxel_um / 10_000.0, 1.5)
                + 2.0 * entropy
                + 1.5 * mean_component_score
                - min(gap_cv, 2.5)
                - 2.0 * mean_fiber
            )
            automated_gate = bool(
                len(glyphs) >= 10
                and span_pixels * voxel_um / 1000.0 >= 5.0
                and gap_cv <= 1.25
                and entropy >= 0.48
                and mean_fiber <= 0.25
                and fiber_dominance_ratio <= 0.35
                and row_parallel_fraction <= 0.55
                and bbox_area_cm2 <= 4.0
            )
            candidates.append(
                {
                    "angle_degrees": angle_degrees,
                    "glyph_proxy_count": len(glyphs),
                    "span_mm": span_pixels * voxel_um / 1000.0,
                    "bbox_width_mm": bbox_width_mm,
                    "bbox_height_mm": bbox_height_mm,
                    "bbox_area_cm2": bbox_area_cm2,
                    "within_four_cm2_axis_aligned_bbox": bbox_area_cm2 <= 4.0,
                    "gap_cv": gap_cv,
                    "orientation_entropy": entropy,
                    "weighted_fiber_likelihood": mean_fiber,
                    "fiber_dominance_ratio": fiber_dominance_ratio,
                    "component_row_parallel_fraction": row_parallel_fraction,
                    "mean_component_score": mean_component_score,
                    "quality": quality,
                    "automated_morphology_gate": automated_gate,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "glyph_proxy_centres_xy": [
                        [round(float(item["x"]), 3), round(float(item["y"]), 3)]
                        for item in glyphs
                    ],
                }
            )
    candidates.sort(key=lambda item: float(item["quality"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        duplicate = False
        for old in selected:
            ox0, oy0, ox1, oy1 = old["bbox_xyxy"]
            intersection = max(0, min(x1, ox1) - max(x0, ox0)) * max(
                0, min(y1, oy1) - max(y0, oy0)
            )
            union = (x1 - x0) * (y1 - y0) + (ox1 - ox0) * (oy1 - oy0) - intersection
            iou = intersection / union if union > 0 else 0.0
            ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
            if iou >= 0.35 or math.hypot(cx - ocx, cy - ocy) < 220:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= top:
            break
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
        # Automated morphology can only nominate review regions.  Legibility
        # and ten actual letters remain explicitly unestablished.
        candidate["defensible_text_candidate"] = False
    return selected


def window_u8(array: np.ndarray, valid: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = array[valid]
    lo, hi = np.percentile(values, (low, high)) if values.size else (0.0, 1.0)
    hi = max(float(hi), float(lo) + 1.0e-6)
    result = np.rint(np.clip((array.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    result[~valid] = 0
    return result


def heat_rgb(array: np.ndarray) -> np.ndarray:
    values = np.clip(array, 0.0, 1.0)
    stops = np.asarray(
        [
            [0.0, 0, 0, 0],
            [0.25, 30, 25, 90],
            [0.50, 190, 35, 55],
            [0.75, 255, 160, 20],
            [1.0, 255, 255, 210],
        ],
        dtype=np.float32,
    )
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.rint(
            np.interp(values, stops[:, 0], stops[:, channel])
        ).astype(np.uint8)
    return rgb


def save_png(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, optimize=True)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def render_candidate_card(
    candidate: Mapping[str, Any],
    center_u8: np.ndarray,
    persistence: np.ndarray,
    fiber: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
) -> Image.Image:
    x0, y0, x1, y1 = candidate_crop_bounds(candidate, score.shape)
    raw, _evidence = render_candidate_native_arrays(
        candidate, center_u8, persistence, fiber, score, valid
    )
    persistence_rgb = heat_rgb(persistence[y0:y1, x0:x1])
    fiber_rgb = heat_rgb(fiber[y0:y1, x0:x1])
    score_rgb = heat_rgb(score[y0:y1, x0:x1])
    alpha = np.clip((score[y0:y1, x0:x1] - 0.25) / 0.75, 0, 1)[..., None] * 0.78
    overlay = np.rint(raw * (1.0 - alpha) + score_rgb * alpha).astype(np.uint8)
    crop_valid = valid[y0:y1, x0:x1]
    for panel in (raw, persistence_rgb, fiber_rgb, overlay):
        panel[~crop_valid] = 0
    labels = ("raw offset 0", "depth-persistent dark", "fiber likelihood", "fiber-suppressed overlay")
    target_width = 520
    label_height = 34
    panels: list[Image.Image] = []
    font = ImageFont.load_default()
    for label, array in zip(labels, (raw, persistence_rgb, fiber_rgb, overlay), strict=True):
        image = Image.fromarray(array)
        height = max(1, round(image.height * target_width / max(image.width, 1)))
        image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (target_width, height + label_height), "black")
        panel.paste(image, (0, label_height))
        ImageDraw.Draw(panel).text((8, 10), label, fill="white", font=font)
        panels.append(panel)
    card_width = sum(panel.width for panel in panels) + 8 * (len(panels) - 1)
    title_height = 52
    card_height = max(panel.height for panel in panels) + title_height
    card = Image.new("RGB", (card_width, card_height), (10, 10, 10))
    title = (
        f"rank {candidate['rank']} bbox=({x0},{y0})-({x1},{y1}) "
        f"proxies={candidate['glyph_proxy_count']} span={candidate['span_mm']:.2f}mm "
        f"fiber={candidate['weighted_fiber_likelihood']:.3f} entropy={candidate['orientation_entropy']:.3f} "
        f"morph_gate={candidate['automated_morphology_gate']}"
    )
    ImageDraw.Draw(card).text((8, 16), title, fill="white", font=font)
    cursor = 0
    for panel in panels:
        card.paste(panel, (cursor, title_height))
        cursor += panel.width + 8
    return card


def shared_depth_window_u8(
    stack: np.ndarray,
    valid: np.ndarray,
    indices: Sequence[int] = REVIEW_DEPTH_INDICES,
) -> tuple[dict[int, np.ndarray], dict[str, float]]:
    """Window selected raw depths together so apparent changes are comparable."""

    values = np.concatenate([stack[index][valid] for index in indices])
    low, high = (float(value) for value in np.percentile(values, (1.0, 99.0)))
    high = max(high, low + 1.0e-6)
    result: dict[int, np.ndarray] = {}
    for index in indices:
        image = np.rint(
            np.clip((stack[index].astype(np.float32) - low) / (high - low), 0, 1)
            * 255
        ).astype(np.uint8)
        image[~valid] = 0
        result[index] = image
    return result, {"low": low, "high": high}


def render_candidate_depth_controls(
    candidate: Mapping[str, Any],
    depth_u8: Mapping[int, np.ndarray],
    persistence: np.ndarray,
    specificity: np.ndarray,
    fiber: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    *,
    voxel_um: float,
) -> Image.Image:
    """Render shared-window raw depths plus explicit fiber/depth controls."""

    x0, y0, x1, y1 = candidate_crop_bounds(candidate, score.shape)
    crop_valid = valid[y0:y1, x0:x1]
    panels: list[tuple[str, np.ndarray]] = []
    for index in REVIEW_DEPTH_INDICES:
        offset = EXACT_OFFSETS_VOXELS[index]
        raw = np.repeat(depth_u8[index][y0:y1, x0:x1, None], 3, axis=2)
        raw[~crop_valid] = 0
        panels.append(
            (f"raw {offset:+d} vox ({offset * voxel_um:+.3f} um)", raw)
        )
    for label, array in (
        ("adjacent-depth persistence", persistence),
        ("surface specificity", specificity),
        ("fiber likelihood (negative control)", fiber),
        ("fiber-suppressed nomination score", score),
    ):
        rgb = heat_rgb(array[y0:y1, x0:x1])
        rgb[~crop_valid] = 0
        panels.append((label, rgb))

    columns = 4
    target_width = 360
    label_height = 32
    font = ImageFont.load_default()
    rendered: list[Image.Image] = []
    for label, array in panels:
        image = Image.fromarray(array)
        height = max(1, round(image.height * target_width / max(image.width, 1)))
        image = image.resize((target_width, height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (target_width, height + label_height), "black")
        panel.paste(image, (0, label_height))
        ImageDraw.Draw(panel).text((7, 10), label, fill="white", font=font)
        rendered.append(panel)
    rows = math.ceil(len(rendered) / columns)
    gap = 7
    cell_height = max(panel.height for panel in rendered)
    title_height = 44
    sheet = Image.new(
        "RGB",
        (
            columns * target_width + (columns - 1) * gap,
            title_height + rows * cell_height + (rows - 1) * gap,
        ),
        (8, 8, 8),
    )
    title = (
        f"rank {candidate['rank']} depth/fiber controls; bbox=({x0},{y0})-({x1},{y1}); "
        "raw panels share one intensity window"
    )
    ImageDraw.Draw(sheet).text((7, 15), title, fill="white", font=font)
    for panel_index, panel in enumerate(rendered):
        column = panel_index % columns
        row = panel_index // columns
        sheet.paste(
            panel,
            (
                column * (target_width + gap),
                title_height + row * (cell_height + gap),
            ),
        )
    return sheet


def candidate_crop_bounds(
    candidate: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in candidate["bbox_xyxy"])
    margin_x, margin_y = 80, 90
    x0 = max(0, x0 - margin_x)
    x1 = min(shape[1], x1 + margin_x)
    y0 = max(0, y0 - margin_y)
    y1 = min(shape[0], y1 + margin_y)
    return x0, y0, x1, y1


def render_candidate_native_arrays(
    candidate: Mapping[str, Any],
    center_u8: np.ndarray,
    persistence: np.ndarray,
    fiber: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = candidate_crop_bounds(candidate, score.shape)
    raw = np.repeat(center_u8[y0:y1, x0:x1, None], 3, axis=2)
    score_rgb = heat_rgb(score[y0:y1, x0:x1])
    alpha = np.clip((score[y0:y1, x0:x1] - 0.25) / 0.75, 0, 1)[..., None] * 0.78
    overlay = np.rint(raw * (1.0 - alpha) + score_rgb * alpha).astype(np.uint8)
    crop_valid = valid[y0:y1, x0:x1]
    for panel in (raw, overlay):
        panel[~crop_valid] = 0
    return raw, overlay


def run_audit(
    stack_dir: Path,
    mask_path: Path,
    output_dir: Path,
    *,
    input_manifest: Path | None = None,
    voxel_um: float = DEFAULT_VOXEL_UM,
    edge_margin_pixels: int = 28,
    background_size: int = 41,
    score_threshold: float = 0.42,
    top_rows: int = 12,
    require_exact_93_manifest: bool = False,
) -> dict[str, Any]:
    if voxel_um <= 0:
        raise IndependentInkAuditError("voxel_um must be positive")
    if edge_margin_pixels < 1:
        raise IndependentInkAuditError("edge margin must be positive")
    if not 0.0 < score_threshold < 1.0:
        raise IndependentInkAuditError("score threshold must be in (0,1)")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The directory is dedicated to this reproducible audit.  Remove only
    # managed per-row images so a rerun with fewer hypotheses cannot leave
    # stale evidence that appears current.
    for stale in output_dir.glob("row_*.png"):
        stale.unlink()
    stack, layer_hashes = load_exact_stack(stack_dir)
    mask = load_mask(mask_path, stack.shape[1:])
    exact_manifest_validation = None
    if require_exact_93_manifest:
        if input_manifest is None:
            raise IndependentInkAuditError(
                "--require-exact-93-manifest requires --input-manifest"
            )
        exact_manifest_validation = validate_exact_93_manifest(
            input_manifest,
            stack_dir,
            layer_hashes,
            stack.shape,
            mask_path,
            mask,
            voxel_um=voxel_um,
        )
    interior = ndimage.distance_transform_edt(mask) >= edge_margin_pixels
    if int(np.count_nonzero(interior)) < 10_000:
        raise IndependentInkAuditError("edge-safe interior is too small for an ink audit")

    selected_indices = tuple(sorted(set(NEAR_INDICES) | set(FAR_INDICES)))
    residuals: dict[int, np.ndarray] = {}
    layer_normalization: dict[str, Any] = {}
    for index in selected_indices:
        residuals[index], layer_normalization[str(index)] = robust_dark_residual(
            stack[index],
            mask,
            interior,
            background_size=background_size,
        )
    near = np.stack([residuals[index] for index in NEAR_INDICES], axis=0)
    far = np.stack([residuals[index] for index in FAR_INDICES], axis=0)
    depth = aggregate_depth_evidence(near, far)
    del residuals, near, far
    normalized_depth, score_window = robust_unit_interval(
        depth["raw_depth_score"], interior
    )
    fiber = fiber_orientation_evidence(normalized_depth, interior)
    independent_score = normalized_depth * (
        1.0 - 0.72 * fiber["fiber_likelihood"]
    )
    independent_score *= 0.35 + 0.65 * depth["near_support"]
    independent_score[~interior] = 0.0
    independent_score, final_window = robust_unit_interval(
        independent_score, interior, low_percentile=70.0, high_percentile=99.5
    )
    persistence_visual = robust_unit_interval(
        depth["adjacent_persistence"],
        interior,
        low_percentile=50,
        high_percentile=99.5,
    )[0]
    far_visual = robust_unit_interval(
        depth["far_persistence"],
        interior,
        low_percentile=50,
        high_percentile=99.5,
    )[0]

    binary = (independent_score >= score_threshold) & interior
    components = component_features(binary, independent_score, fiber["fiber_likelihood"])
    candidates = rank_row_hypotheses(
        components,
        independent_score,
        fiber["fiber_likelihood"],
        fiber["orientation_radians"],
        interior,
        voxel_um=voxel_um,
        top=top_rows,
    )

    float_maps = {
        "adjacent_depth_persistence.tif": depth["adjacent_persistence"],
        "far_depth_persistence.tif": depth["far_persistence"],
        "near_depth_support.tif": depth["near_support"],
        "surface_specificity.tif": depth["surface_specificity"],
        "fiber_ridge_likelihood.tif": fiber["ridge_fiber_likelihood"],
        "fiber_likelihood.tif": fiber["fiber_likelihood"],
        "independent_ink_score.tif": independent_score,
    }
    artifacts: dict[str, Any] = {}
    for filename, array in float_maps.items():
        path = output_dir / filename
        tifffile.imwrite(path, array.astype(np.float32))
        artifacts[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "dtype": "float32",
            "shape": list(array.shape),
            "minimum_interior": float(np.min(array[interior])),
            "maximum_interior": float(np.max(array[interior])),
            "mean_interior": float(np.mean(array[interior])),
        }

    center_u8 = window_u8(stack[CENTER_INDEX], mask)
    global_pngs = {
        "center_offset0.png": center_u8,
        "adjacent_depth_persistence.png": heat_rgb(persistence_visual),
        "far_depth_persistence.png": heat_rgb(far_visual),
        "surface_specificity.png": heat_rgb(depth["surface_specificity"]),
        "fiber_ridge_likelihood.png": heat_rgb(fiber["ridge_fiber_likelihood"]),
        "fiber_likelihood.png": heat_rgb(fiber["fiber_likelihood"]),
        "independent_ink_score.png": heat_rgb(independent_score),
    }
    for filename, image in global_pngs.items():
        image = image.copy()
        image[~mask] = 0
        artifacts[filename] = save_png(output_dir / filename, image)

    component_path = output_dir / "components.tsv"
    component_fields = [
        "label", "x", "y", "x0", "y0", "x1", "y1", "width", "height",
        "area", "skeleton_length", "elongation", "mean_score", "peak_score",
        "principal_angle_degrees", "mean_fiber_likelihood", "long_thin_fiber_proxy",
    ]
    with component_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=component_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(components)
    artifacts[component_path.name] = {
        "path": str(component_path.resolve()),
        "sha256": sha256_file(component_path),
        "bytes": component_path.stat().st_size,
    }

    candidates_path = output_dir / "ranked_row_hypotheses.json"
    atomic_json(candidates_path, {"schema_version": 1, "candidates": candidates})
    artifacts[candidates_path.name] = {
        "path": str(candidates_path.resolve()),
        "sha256": sha256_file(candidates_path),
        "bytes": candidates_path.stat().st_size,
    }
    depth_u8, shared_depth_window = shared_depth_window_u8(stack, mask)
    cards: list[Image.Image] = []
    for candidate in candidates:
        card = render_candidate_card(
            candidate,
            center_u8,
            persistence_visual,
            fiber["fiber_likelihood"],
            independent_score,
            mask,
        )
        filename = f"row_{candidate['rank']:02d}.png"
        path = output_dir / filename
        card.save(path, optimize=True)
        artifacts[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        native_raw, native_evidence = render_candidate_native_arrays(
            candidate,
            center_u8,
            persistence_visual,
            fiber["fiber_likelihood"],
            independent_score,
            mask,
        )
        for native_role, native_array in (
            ("native_raw", native_raw),
            ("native_evidence", native_evidence),
        ):
            native_filename = f"row_{candidate['rank']:02d}_{native_role}.png"
            artifacts[native_filename] = save_png(
                output_dir / native_filename, native_array
            )
        depth_controls = render_candidate_depth_controls(
            candidate,
            depth_u8,
            persistence_visual,
            depth["surface_specificity"],
            fiber["fiber_likelihood"],
            independent_score,
            mask,
            voxel_um=voxel_um,
        )
        depth_filename = f"row_{candidate['rank']:02d}_depth_controls.png"
        depth_path = output_dir / depth_filename
        depth_controls.save(depth_path, optimize=True)
        artifacts[depth_filename] = {
            "path": str(depth_path.resolve()),
            "sha256": sha256_file(depth_path),
            "bytes": depth_path.stat().st_size,
            "raw_frame_indices": list(REVIEW_DEPTH_INDICES),
            "raw_offsets_voxels": [
                EXACT_OFFSETS_VOXELS[index] for index in REVIEW_DEPTH_INDICES
            ],
            "shared_raw_intensity_window": shared_depth_window,
        }
        cards.append(card)
    if cards:
        gap = 10
        sheet = Image.new(
            "RGB",
            (max(card.width for card in cards), sum(card.height for card in cards) + gap * (len(cards) - 1)),
            "black",
        )
        cursor = 0
        for card in cards:
            sheet.paste(card, (0, cursor))
            cursor += card.height + gap
        contact_path = output_dir / "ranked_rows_contact_sheet.png"
        sheet.save(contact_path, optimize=True)
        artifacts[contact_path.name] = {
            "path": str(contact_path.resolve()),
            "sha256": sha256_file(contact_path),
            "bytes": contact_path.stat().st_size,
        }

    input_manifest_record = None
    if input_manifest is not None:
        if not input_manifest.is_file():
            raise IndependentInkAuditError(f"input manifest is missing: {input_manifest}")
        input_manifest_record = {
            "path": str(input_manifest.resolve()),
            "sha256": sha256_file(input_manifest),
            "exact_93_layer_contract": exact_manifest_validation,
        }
    layer_index_digest = hashlib.sha256(
        json.dumps(layer_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    gate_passes = [candidate for candidate in candidates if candidate["automated_morphology_gate"]]
    report = {
        "schema_version": 1,
        "status": "complete",
        "role": "model_independent_raw_stack_triage_not_text_or_prize_validation",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
            "tests_path": str(
                (Path(__file__).parent / "tests" / "test_independent_raw_ink_audit.py").resolve()
            ),
            "tests_sha256": sha256_file(
                Path(__file__).parent / "tests" / "test_independent_raw_ink_audit.py"
            ),
        },
        "method": {
            "learned_models_used": [],
            "model_probability_maps_used": [],
            "network_services_used": [],
            "gpu_used": False,
            "cost_usd": 0.0,
            "contrast_polarity": "dark relative to a mask-normalized local sheet baseline",
            "near_blocks_half_open": [list(block) for block in NEAR_BLOCKS],
            "far_control_indices": list(FAR_INDICES),
            "manual_review_depth_indices": list(REVIEW_DEPTH_INDICES),
            "manual_review_depth_offsets_voxels": [
                EXACT_OFFSETS_VOXELS[index] for index in REVIEW_DEPTH_INDICES
            ],
            "row_baseline_angle_degrees": list(ROW_ANGLE_DEGREES),
            "maximum_row_context_millimeters": MAXIMUM_ROW_CONTEXT_UM / 1000.0,
            "depth_persistence": "maximum minimum of adjacent seven-layer block medians",
            "surface_specificity": "adjacent / (adjacent + 0.75 * far_median + 0.5)",
            "fiber_suppression": (
                "maximum of 24-pixel structure-tensor directional evidence and "
                "multi-scale 1..4-pixel Frangi ridge evidence"
            ),
            "row_gate_is_only_morphological": True,
        },
        "parameters": {
            "voxel_um": voxel_um,
            "edge_margin_pixels": edge_margin_pixels,
            "background_size_pixels": background_size,
            "score_threshold": score_threshold,
            "top_rows": top_rows,
            "require_exact_93_manifest": require_exact_93_manifest,
            "shared_manual_review_raw_window": shared_depth_window,
            "score_window": score_window,
            "final_score_window": final_window,
        },
        "inputs": {
            "stack_directory": str(stack_dir.resolve()),
            "stack_shape_zyx": list(stack.shape),
            "stack_layer_sha256_index_sha256": layer_index_digest,
            "mask": {
                "path": str(mask_path.resolve()),
                "sha256": sha256_file(mask_path),
                "valid_pixels": int(np.count_nonzero(mask)),
                "interior_pixels": int(np.count_nonzero(interior)),
            },
            "input_manifest": input_manifest_record,
        },
        "layer_normalization": layer_normalization,
        "statistics": {
            "threshold_pixels": int(np.count_nonzero(binary)),
            "threshold_fraction_of_interior": float(np.mean(binary[interior])),
            "components": len(components),
            "long_thin_fiber_proxy_components": sum(
                bool(component["long_thin_fiber_proxy"]) for component in components
            ),
            "ranked_row_hypotheses": len(candidates),
            "automated_morphology_gate_passes": len(gate_passes),
            "defensible_text_candidates": 0,
        },
        "verdict": {
            "automated_morphology_gate_pass": bool(gate_passes),
            "visible_legible_letters_established": False,
            "ten_letter_criterion_established": False,
            "defensible_candidate_count_before_manual_review": 0,
            "reason": (
                "Raw morphology can prioritize review, but cannot establish legible letters. "
                "Every hypothesis remains fail-closed until its raw panels are manually reviewed "
                "for fiber/crack explanations and actual character sequences."
            ),
        },
        "candidates": candidates,
        "artifacts": artifacts,
    }
    report_path = output_dir / "audit-report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stack_dir", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--voxel-um", type=float, default=DEFAULT_VOXEL_UM)
    parser.add_argument("--edge-margin-pixels", type=int, default=28)
    parser.add_argument("--background-size", type=int, default=41)
    parser.add_argument("--score-threshold", type=float, default=0.42)
    parser.add_argument("--top-rows", type=int, default=12)
    parser.add_argument("--require-exact-93-manifest", action="store_true")
    args = parser.parse_args()
    run_audit(
        args.stack_dir,
        args.mask,
        args.output_dir,
        input_manifest=args.input_manifest,
        voxel_um=args.voxel_um,
        edge_margin_pixels=args.edge_margin_pixels,
        background_size=args.background_size,
        score_threshold=args.score_threshold,
        top_rows=args.top_rows,
        require_exact_93_manifest=args.require_exact_93_manifest,
    )


if __name__ == "__main__":
    main()
