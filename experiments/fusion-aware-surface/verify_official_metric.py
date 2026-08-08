#!/usr/bin/env python3
"""Smoke-test the source-pinned official topology-aware metric."""

from __future__ import annotations

import json

import numpy as np

from official_metric import compute_official


def main() -> int:
    surface = np.zeros((24, 24, 24), dtype=np.uint8)
    surface[12, :, :] = 1
    result = compute_official(surface, surface.copy())
    if set(result) != {"blend", "toposcore", "surface_dice", "voi_score"}:
        raise RuntimeError("official metric returned the wrong schema")
    if not all(abs(value - 1.0) < 1e-12 for value in result.values()):
        raise RuntimeError(f"identity surface did not score one: {result}")
    print(json.dumps(result, sort_keys=True))
    print("OFFICIAL_METRIC_IDENTITY_CHECK_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
