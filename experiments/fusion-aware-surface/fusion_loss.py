"""Binary surface loss with an isolated exact-gap weighting intervention."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fusion_aware_surface_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gap_mask: torch.Tensor,
    *,
    gap_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return CE + soft Dice; optionally up-weight exact background gaps.

    The control arm calls this function with `gap_weight=1`. The intervention
    calls it with the preregistered value 8. Every other operation is shared.
    """
    if logits.ndim < 4 or logits.shape[1] != 2:
        raise ValueError("logits must have shape N,2,...")
    expected = (logits.shape[0],) + tuple(logits.shape[2:])
    if tuple(target.shape) != expected or tuple(gap_mask.shape) != expected:
        raise ValueError(
            f"target and gap_mask must have shape {expected}; got "
            f"{tuple(target.shape)} and {tuple(gap_mask.shape)}"
        )
    if gap_weight < 1:
        raise ValueError("gap_weight must be at least one")

    target_long = target.to(dtype=torch.long)
    if torch.any((target_long != 0) & (target_long != 1)):
        raise ValueError("target must be binary")
    gap = gap_mask.to(dtype=torch.bool)
    if torch.any(gap & (target_long != 0)):
        raise ValueError("gap supervision may mark background voxels only")

    ce_voxel = F.cross_entropy(logits.float(), target_long, reduction="none")
    weight = torch.ones_like(ce_voxel)
    if gap_weight != 1:
        weight = torch.where(gap, torch.as_tensor(gap_weight, device=weight.device), weight)
    weighted_ce = (ce_voxel * weight).sum() / weight.sum().clamp_min(1)

    probability = torch.softmax(logits.float(), dim=1)[:, 1]
    target_float = target_long.to(dtype=probability.dtype)
    intersection = (probability * target_float).sum()
    dice_loss = 1 - (2 * intersection + 1) / (
        probability.sum() + target_float.sum() + 1
    )
    total = weighted_ce + dice_loss
    return total, {
        "weighted_ce": weighted_ce.detach(),
        "dice": dice_loss.detach(),
        "gap_voxels": gap.sum().detach(),
    }

