"""Fail-closed CT normalization for the pinned Dataset059 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EXPECTED_CT_PROPERTIES = {
    "mean": 129.09779357910156,
    "std": 42.188316345214844,
    "percentile_00_5": 44.0,
    "percentile_99_5": 236.0,
}


def validate_ct_contract(
    scheme: object,
    properties: Mapping[str, object] | None,
    *,
    tolerance: float = 1e-3,
) -> dict[str, float]:
    """Return numeric CT properties only when the checkpoint contract matches."""
    normalized_scheme = str(scheme).lower().replace("_", "")
    if "ctnormalization" not in normalized_scheme and normalized_scheme != "ct":
        raise ValueError(f"expected CTNormalization, received {scheme!r}")
    if properties is None:
        raise ValueError("checkpoint is missing intensity_properties")

    values: dict[str, float] = {}
    for key, expected in EXPECTED_CT_PROPERTIES.items():
        if key not in properties:
            raise ValueError(f"checkpoint intensity_properties missing {key!r}")
        value = float(properties[key])
        if not np.isfinite(value):
            raise ValueError(f"checkpoint intensity property {key!r} is non-finite")
        if abs(value - expected) > tolerance:
            raise ValueError(
                f"checkpoint intensity property {key!r} changed: "
                f"expected {expected}, received {value}"
            )
        values[key] = value
    if values["std"] <= 0:
        raise ValueError("checkpoint CT standard deviation must be positive")
    if values["percentile_00_5"] >= values["percentile_99_5"]:
        raise ValueError("checkpoint CT clipping interval is empty")
    return values


def ct_normalize(image: np.ndarray, properties: Mapping[str, float]) -> np.ndarray:
    """Apply nnU-Net CTNormalization: clip, then checkpoint mean/std."""
    lo = float(properties["percentile_00_5"])
    hi = float(properties["percentile_99_5"])
    mean = float(properties["mean"])
    std = float(properties["std"])
    value = np.asarray(image, dtype=np.float32)
    value = np.clip(value, lo, hi)
    return (value - mean) / (std + 1e-8)
