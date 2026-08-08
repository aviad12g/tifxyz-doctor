import pytest
import torch

from fusion_loss import fusion_aware_surface_loss


def fixture():
    logits = torch.zeros((1, 2, 2, 2, 3), dtype=torch.float32)
    target = torch.zeros((1, 2, 2, 3), dtype=torch.long)
    target[:, :, :, 0] = 1
    gap = torch.zeros_like(target, dtype=torch.bool)
    gap[:, :, :, 1] = True
    return logits, target, gap


def test_control_and_gap_arm_share_loss_when_no_gap_exists() -> None:
    logits, target, gap = fixture()
    gap.zero_()
    control, _ = fusion_aware_surface_loss(logits, target, gap, gap_weight=1)
    intervention, _ = fusion_aware_surface_loss(logits, target, gap, gap_weight=8)
    assert torch.equal(control, intervention)


def test_gap_weight_increases_gap_background_gradient() -> None:
    logits_a, target, gap = fixture()
    logits_b = logits_a.clone()
    logits_a.requires_grad_()
    logits_b.requires_grad_()
    control, _ = fusion_aware_surface_loss(logits_a, target, gap, gap_weight=1)
    intervention, report = fusion_aware_surface_loss(
        logits_b, target, gap, gap_weight=8
    )
    control.backward()
    intervention.backward()
    control_gap_grad = logits_a.grad[:, 1][gap].abs().mean()
    intervention_gap_grad = logits_b.grad[:, 1][gap].abs().mean()
    assert intervention_gap_grad > control_gap_grad
    assert int(report["gap_voxels"]) == int(gap.sum())


def test_nearly_correct_logits_have_small_loss() -> None:
    logits, target, gap = fixture()
    logits[:, 0] = 8
    logits[:, 1] = -8
    logits[:, 0][target == 1] = -8
    logits[:, 1][target == 1] = 8
    loss, _ = fusion_aware_surface_loss(logits, target, gap, gap_weight=8)
    assert float(loss) < 1e-4


def test_rejects_foreground_gap_and_invalid_weight() -> None:
    logits, target, gap = fixture()
    bad = gap.clone()
    bad[target == 1] = True
    with pytest.raises(ValueError):
        fusion_aware_surface_loss(logits, target, bad, gap_weight=8)
    with pytest.raises(ValueError):
        fusion_aware_surface_loss(logits, target, gap, gap_weight=0.5)


def test_rejects_shape_mismatch() -> None:
    logits, target, gap = fixture()
    with pytest.raises(ValueError):
        fusion_aware_surface_loss(logits, target[..., :-1], gap, gap_weight=8)

