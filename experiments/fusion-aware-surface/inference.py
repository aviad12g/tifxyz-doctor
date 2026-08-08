"""Frozen sliding-window inference for the fusion-aware experiment."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from normalization import ct_normalize
from train_fusion_aware import extract_logits


PATCH_SIZE = 192
STEP_SIZE = 96
GAUSSIAN_SIGMA_SCALE = 1.0 / 8.0
MIN_WEIGHT = 1e-4


def sliding_starts(length: int, patch: int = PATCH_SIZE, step: int = STEP_SIZE) -> list[int]:
    if length < 1 or patch < 1 or step < 1:
        raise ValueError("length, patch, and step must be positive")
    if length <= patch:
        return [0]
    values = list(range(0, length - patch + 1, step))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def gaussian_weight(patch: int = PATCH_SIZE) -> np.ndarray:
    coordinate = np.arange(patch, dtype=np.float32) - (patch - 1) / 2
    sigma = patch * GAUSSIAN_SIGMA_SCALE
    one_dimensional = np.exp(-0.5 * np.square(coordinate / sigma))
    weight = (
        one_dimensional[:, None, None]
        * one_dimensional[None, :, None]
        * one_dimensional[None, None, :]
    )
    weight /= weight.max()
    return np.maximum(weight, MIN_WEIGHT).astype(np.float32)


@torch.inference_mode()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    properties: Mapping[str, float],
    device: torch.device,
    *,
    patch: int = PATCH_SIZE,
    step: int = STEP_SIZE,
) -> np.ndarray:
    """Return float32 surface probability with frozen Gaussian blending."""
    normalized = ct_normalize(image, properties)
    if normalized.ndim != 3:
        raise ValueError(f"image must be ZYX, got {normalized.shape}")
    output = np.zeros(normalized.shape, dtype=np.float32)
    denominator = np.zeros(normalized.shape, dtype=np.float32)
    weight = gaussian_weight(patch)
    starts = [sliding_starts(length, patch, step) for length in normalized.shape]
    model.eval()

    for z0 in starts[0]:
        for y0 in starts[1]:
            for x0 in starts[2]:
                cube = normalized[z0 : z0 + patch, y0 : y0 + patch, x0 : x0 + patch]
                valid_shape = cube.shape
                padding = [(0, patch - length) for length in valid_shape]
                if any(after for _, after in padding):
                    cube = np.pad(cube, padding, mode="reflect")
                tensor = torch.from_numpy(np.ascontiguousarray(cube))[None, None].to(device)
                with torch.amp.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
                    logits = extract_logits(model(tensor))
                probability = torch.softmax(logits.float(), dim=1)[0, 1]
                probability = probability.cpu().numpy()[
                    : valid_shape[0], : valid_shape[1], : valid_shape[2]
                ]
                local_weight = weight[
                    : valid_shape[0], : valid_shape[1], : valid_shape[2]
                ]
                region = np.s_[
                    z0 : z0 + valid_shape[0],
                    y0 : y0 + valid_shape[1],
                    x0 : x0 + valid_shape[2],
                ]
                output[region] += probability * local_weight
                denominator[region] += local_weight
                del tensor, logits, probability

    if np.any(denominator == 0):
        raise RuntimeError("sliding-window inference left uncovered voxels")
    result = output / denominator
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise RuntimeError("invalid probability output")
    return result
