import numpy as np
import pytest
import torch

from inference import gaussian_weight, predict_volume, sliding_starts
from normalization import EXPECTED_CT_PROPERTIES


def test_sliding_starts_covers_end_exactly_once() -> None:
    assert sliding_starts(192) == [0]
    assert sliding_starts(300) == [0, 96, 108]
    assert sliding_starts(384) == [0, 96, 192]


def test_gaussian_weight_contract() -> None:
    weight = gaussian_weight(16)
    assert weight.shape == (16, 16, 16)
    assert weight.dtype == np.float32
    assert float(weight.max()) == pytest.approx(1.0)
    assert float(weight.min()) >= float(np.float32(1e-4))


def test_constant_model_survives_overlap_blending() -> None:
    class Constant(torch.nn.Module):
        def forward(self, value):
            shape = (value.shape[0], 2) + tuple(value.shape[2:])
            return torch.zeros(shape, dtype=value.dtype, device=value.device)

    image = np.full((12, 12, 12), 129, dtype=np.uint8)
    probability = predict_volume(
        Constant(),
        image,
        EXPECTED_CT_PROPERTIES,
        torch.device("cpu"),
        patch=8,
        step=4,
    )
    assert probability.shape == image.shape
    assert np.allclose(probability, 0.5, atol=1e-6)
