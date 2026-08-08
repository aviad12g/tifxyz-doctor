from pathlib import Path

import pytest

from official_metric import METRIC_DATASET, METRIC_ZIP_SHA256, verify_metric_source


def test_official_metric_pin_is_explicit() -> None:
    assert METRIC_DATASET == "sohier/vesuvius-metric-resources"
    assert len(METRIC_ZIP_SHA256) == 64


def test_official_metric_source_check_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="source mismatch"):
        verify_metric_source(tmp_path)
