import numpy as np
import pytest

from gap_supervision import broadcast_gap_mask, inter_sheet_gap_mask


def parallel_sheets() -> np.ndarray:
    labels = np.zeros((11, 13), dtype=np.int16)
    labels[:, 3] = 11
    labels[:, 7] = 29
    return labels


def test_marks_only_background_supported_by_two_instances() -> None:
    labels = parallel_sheets()
    gap = inter_sheet_gap_mask(labels, radius=2)
    assert gap[:, 5].all()
    assert not gap[:, :5].any()
    assert not gap[:, 6:].any()
    assert not gap[labels > 0].any()


def test_single_sheet_has_no_inter_sheet_gap() -> None:
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[:, 4] = 3
    assert not inter_sheet_gap_mask(labels, radius=3).any()


def test_join_only_surface_is_excluded() -> None:
    labels = parallel_sheets()
    surface = labels > 0
    surface[:, 5] = True
    gap = inter_sheet_gap_mask(labels, surface, radius=2)
    assert not gap[:, 5].any()


def test_relabeling_does_not_change_gap_geometry() -> None:
    labels = parallel_sheets()
    relabeled = labels.copy()
    relabeled[labels == 11] = 2
    relabeled[labels == 29] = 999
    assert np.array_equal(
        inter_sheet_gap_mask(labels, radius=2),
        inter_sheet_gap_mask(relabeled, radius=2),
    )


def test_broadcast_is_zyx_and_writable() -> None:
    mask = np.eye(4, dtype=bool)
    volume = broadcast_gap_mask(mask, 3)
    assert volume.shape == (3, 4, 4)
    assert volume.flags.writeable
    volume[0, 0, 0] = False
    assert volume[1, 0, 0]


def test_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        inter_sheet_gap_mask(np.zeros((2, 2, 2), dtype=np.int16))
    with pytest.raises(TypeError):
        inter_sheet_gap_mask(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        inter_sheet_gap_mask(np.array([[0, -1]], dtype=np.int16))
    labels = np.array([[0, 1]], dtype=np.int16)
    with pytest.raises(ValueError):
        inter_sheet_gap_mask(labels, np.zeros_like(labels), radius=1)

