"""Losses local to transformer-family baselines.

The baseline objective intentionally does not reuse ``src.losses``. It matches
the effective non-zero configuration:

    PIT plain L1 + lambda_activity * activity BCE

Optionally, callers can pass a mixture reference and non-zero ``lambda_mix`` to
add a soft mixture-sum L1 term for direct-prediction baselines.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def _match_source_count(pred: Tensor, target: Tensor) -> Tensor:
    num_sources = pred.shape[1]
    if target.shape[1] == num_sources:
        return target
    if target.shape[1] > num_sources:
        return target[:, :num_sources]

    pad_n = num_sources - target.shape[1]
    padding = torch.zeros(
        target.shape[0],
        pad_n,
        target.shape[2],
        dtype=target.dtype,
        device=target.device,
    )
    return torch.cat([target, padding], dim=1)


def calculate_pit_loss(pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Permutation-invariant plain L1 source separation loss."""
    batch_size, num_sources, _ = pred.shape
    target = _match_source_count(pred, target)
    pairwise_l1 = torch.abs(pred.unsqueeze(2) - target.unsqueeze(1)).mean(dim=-1)

    perms_tensor = torch.tensor(list(permutations(range(num_sources))), device=pred.device)
    num_perms = perms_tensor.shape[0]
    losses_expanded = pairwise_l1.unsqueeze(1).expand(-1, num_perms, -1, -1)
    gather_indices = perms_tensor.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
    perm_losses = torch.gather(losses_expanded, 3, gather_indices).squeeze(-1).sum(dim=2)

    min_loss, min_indices = torch.min(perm_losses, dim=1)
    best_perms = perms_tensor[min_indices]
    return min_loss.mean(), best_perms


def calculate_activity_loss(
    activity_logits: Tensor,
    targets: Tensor,
    best_perms: Tensor,
) -> tuple[Tensor, Tensor]:
    """BCE on active source slots after applying the best PIT permutation."""
    targets = _match_source_count(activity_logits.unsqueeze(-1), targets)
    target_energy = (targets**2).sum(dim=-1)
    target_is_active = (target_energy > 1e-6).float()
    aligned_activity = torch.gather(target_is_active, 1, best_perms)
    activity_loss = F.binary_cross_entropy_with_logits(activity_logits, aligned_activity)
    return activity_loss, aligned_activity


def calculate_mixture_l1_loss(pred: Tensor, mixture_ref: Tensor) -> Tensor:
    """Plain L1 between summed predicted sources and the input mixture."""
    if mixture_ref.dim() == 3:
        if mixture_ref.shape[1] != 1:
            raise ValueError(f"Expected mixture_ref [B, 1, L], got {mixture_ref.shape}")
        mixture_ref = mixture_ref.squeeze(1)
    return torch.abs(pred.sum(dim=1) - mixture_ref).mean()


def calculate_baseline_loss(
    pred: Tensor,
    target: Tensor,
    activity_logits: Tensor,
    lambda_activity: float = 2.0,
    mixture_ref: Tensor | None = None,
    lambda_mix: float = 0.0,
) -> tuple[Tensor, Tensor, Dict[str, Tensor], Tensor]:
    """Baseline objective: PIT plain L1 + activity BCE + optional mixture L1."""
    sep_loss, best_perms = calculate_pit_loss(pred, target)
    activity_loss, aligned_activity = calculate_activity_loss(activity_logits, target, best_perms)
    total_loss = sep_loss + lambda_activity * activity_loss
    loss_parts = {"sep_loss": sep_loss, "activity_loss": activity_loss}
    if lambda_mix > 0 and mixture_ref is not None:
        mixture_loss = calculate_mixture_l1_loss(pred, mixture_ref)
        total_loss = total_loss + lambda_mix * mixture_loss
        loss_parts["mixture_loss"] = mixture_loss
    return total_loss, best_perms, loss_parts, aligned_activity


def calculate_pairwise_sisdr(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Pairwise SI-SDR for reporting only."""
    target = _match_source_count(pred, target)
    pred_exp = pred.unsqueeze(2)
    target_exp = target.unsqueeze(1)

    dot = torch.sum(pred_exp * target_exp, dim=-1)
    target_energy = torch.sum(target_exp**2, dim=-1).clamp_min(eps)
    scale = dot / target_energy
    projected = scale.unsqueeze(-1) * target_exp
    residual = pred_exp - projected

    signal_energy = torch.sum(projected**2, dim=-1)
    noise_energy = torch.sum(residual**2, dim=-1).clamp_min(eps)
    sisdr = 10 * torch.log10((signal_energy + eps) / noise_energy)
    target_is_silent = target_energy < 1e-6
    return torch.where(target_is_silent.expand_as(sisdr), torch.zeros_like(sisdr), sisdr)


def calculate_pit_sisdr(pred: Tensor, target: Tensor) -> float:
    """Average best-permutation SI-SDR for validation reporting."""
    batch_size, num_sources, _ = pred.shape
    pairwise_sisdr = calculate_pairwise_sisdr(pred, target)

    perms_tensor = torch.tensor(list(permutations(range(num_sources))), device=pred.device)
    num_perms = perms_tensor.shape[0]
    sisdr_expanded = pairwise_sisdr.unsqueeze(1).expand(-1, num_perms, -1, -1)
    gather_indices = perms_tensor.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
    perm_sisdr = torch.gather(sisdr_expanded, 3, gather_indices).squeeze(-1).mean(dim=2)

    best_sisdr, _ = torch.max(perm_sisdr, dim=1)
    return best_sisdr.mean().item()
