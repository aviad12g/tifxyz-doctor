import numpy as np
import torch

from train_fusion_aware import (
    PAPYRUS,
    PITCHES,
    TRAIN_SEEDS,
    configure_decoder_only,
    crop_arrays,
)


def test_synthetic_training_map_is_a_complete_factorial() -> None:
    combinations = [(pitch, papyrus) for pitch in PITCHES for papyrus in PAPYRUS]
    assert len(TRAIN_SEEDS) == len(combinations) == 16
    assert len(set(combinations)) == 16


def test_crop_keeps_paired_arrays_aligned() -> None:
    base = np.arange(170 * 171 * 172, dtype=np.int32).reshape(170, 171, 172)
    rng = np.random.default_rng(23)
    a, b = crop_arrays((base, base + 9), rng)
    assert a.shape == b.shape == (160, 160, 160)
    assert np.array_equal(b - a, np.full_like(a, 9))


def test_decoder_only_partition() -> None:
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_encoder = torch.nn.Linear(2, 2)
            self.task_decoders = torch.nn.ModuleDict({"surface": torch.nn.Linear(2, 2)})

    model = Toy()
    trainable = configure_decoder_only(model)
    assert trainable == sum(p.numel() for p in model.task_decoders.parameters())
    assert not any(p.requires_grad for p in model.shared_encoder.parameters())
    assert all(p.requires_grad for p in model.task_decoders.parameters())
