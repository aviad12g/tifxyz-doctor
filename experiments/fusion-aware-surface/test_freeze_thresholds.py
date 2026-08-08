from freeze_thresholds import choose_threshold


def test_threshold_selection_maximizes_blend() -> None:
    assert choose_threshold({0.3: 0.4, 0.5: 0.6, 0.7: 0.5}) == 0.5


def test_threshold_tie_prefers_nearest_half_then_lower() -> None:
    assert choose_threshold({0.3: 0.6, 0.4: 0.6, 0.6: 0.6, 0.7: 0.6}) == 0.4
