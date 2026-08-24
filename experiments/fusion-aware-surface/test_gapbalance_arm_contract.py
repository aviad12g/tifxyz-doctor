import pytest

import kaggle_train_runner
import train_fusion_aware


EXPECTED = {
    "control": 1.0,
    "gap2": 2.0,
    "gap4": 4.0,
    "gap8": 8.0,
}


def test_gapbalance_extends_but_does_not_change_original_arm_weights():
    assert train_fusion_aware.GAP_WEIGHTS == EXPECTED
    assert kaggle_train_runner.GAP_WEIGHTS == EXPECTED
    assert train_fusion_aware.gap_weight_for_arm("control") == 1.0
    assert train_fusion_aware.gap_weight_for_arm("gap8") == 8.0


def test_only_gap_weight_changes_across_gapbalance_arms():
    assert train_fusion_aware.gap_weight_for_arm("gap2") == 2.0
    assert train_fusion_aware.gap_weight_for_arm("gap4") == 4.0
    assert set(train_fusion_aware.GAP_WEIGHTS) == set(kaggle_train_runner.GAP_WEIGHTS)


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="unsupported gap-supervision arm"):
        train_fusion_aware.gap_weight_for_arm("gap3")
