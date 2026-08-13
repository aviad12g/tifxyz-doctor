#!/usr/bin/env python3
"""Tests for joint collection of real v5 and retained synthetic v4."""

from __future__ import annotations

import collect_mixed_scoring_results as mixed


def test_package_by_mode_selects_exact_record() -> None:
    records = [{"mode": "real"}, {"mode": "synthetic"}]
    assert mixed.package_by_mode(records, "real") is records[0]
    assert mixed.package_by_mode(records, "synthetic") is records[1]
