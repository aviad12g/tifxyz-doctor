#!/usr/bin/env python3
"""Tests for the result-blind real-scorer CPU metadata adapter."""

from __future__ import annotations

import json
from pathlib import Path


def test_kaggle_cpu_metadata_is_explicitly_accelerator_free() -> None:
    source = Path(__file__).with_name("adapt_real_scoring_cpu_metadata.py").read_text(
        encoding="utf-8"
    )
    assert 'metadata["enable_gpu"] = False' in source
    assert 'metadata["machine_shape"] = ""' in source
    assert '"scorer_source_changed": False' in source
    assert '"scientific_input_or_setting_changed": False' in source
