import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    """Calculate Pearson correlation."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    pred_mean = pred_pattern.mean(dim=-1, keepdim=True)
    target_mean = target_pattern.mean(dim=-1, keepdim=True)
    pred_centered = pred_pattern - pred_mean
    target_centered = target_pattern - target_mean
    numerator = (pred_centered * target_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)
    return (numerator / (pred_std * target_std + 1e-8)).mean().item()


def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculate SI-SDR in dB."""
    if pred.ndim > 2:
        length = pred.shape[-1]
        pred = pred.reshape(-1, length)
        target = target.reshape(-1, length)
    elif pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    dot_product = torch.sum(pred * target, dim=-1)
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    alpha = dot_product / target_energy
    e_target = alpha.unsqueeze(-1) * target
    e_res = pred - e_target
    signal_energy = torch.sum(e_target ** 2, dim=-1)
    noise_energy = torch.sum(e_res ** 2, dim=-1) + eps
    ratio = torch.clamp(signal_energy / noise_energy, min=1e-10)
    return (10 * torch.log10(ratio)).mean().item()


def calculate_sir_sar(
    pred: torch.Tensor,
    target: torch.Tensor,
    interference: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """Calculate SIR and SAR."""
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    artifacts = pred - target
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    artifacts_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)

    if interference is None:
        interference = artifacts
    elif interference.ndim == 1:
        interference = interference.unsqueeze(0)

    interference_energy = torch.sum(interference ** 2, dim=-1) + eps
    sir = 10 * torch.log10(target_energy / interference_energy)
    return sir.mean().item(), sar.mean().item()


def find_xrd_peaks_batch(
    xrd_patterns: torch.Tensor,
    height_threshold: float = 0.05,
    distance: int = 10,
) -> torch.Tensor:
    """Detect peaks in batched XRD patterns."""
    if xrd_patterns.dim() == 1:
        xrd_patterns = xrd_patterns.unsqueeze(0)

    xrd_patterns = xrd_patterns.float()
    mask = xrd_patterns > height_threshold
    x_pad = F.pad(xrd_patterns, (1, 1), value=-1e9)
    is_local_max = (xrd_patterns > x_pad[:, :-2]) & (xrd_patterns > x_pad[:, 2:])
    mask = mask & is_local_max

    if distance > 0:
        kernel_size = 2 * distance + 1
        x_pad_pool = F.pad(xrd_patterns.unsqueeze(1), (distance, distance), value=-1e9)
        max_in_window = F.max_pool1d(x_pad_pool, kernel_size=kernel_size, stride=1).squeeze(1)
        mask = mask & (torch.abs(xrd_patterns - max_in_window) < 1e-6)

    return mask


def _get_batch_peak_indices(mask: torch.Tensor) -> torch.Tensor:
    batch_size, length = mask.shape
    device = mask.device
    idx_map = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, length)
    masked_idxs = torch.where(mask, idx_map, torch.tensor(length, device=device))
    sorted_idxs, _ = torch.sort(masked_idxs, dim=1)
    max_peaks = mask.sum(dim=1).max().item()
    if max_peaks == 0:
        return torch.full((batch_size, 0), -1, device=device, dtype=torch.long)

    dense_idxs = sorted_idxs[:, :max_peaks].clone()
    dense_idxs[dense_idxs == length] = -1
    return dense_idxs


def calculate_peak_shift_delta_2theta(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    two_theta_range: Tuple[float, float] = (5.0, 90.0),
    tolerance: int = 10,
    **peak_args: Any,
) -> float:
    """Calculate mean peak shift in delta 2-theta."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    _, length = pred_pattern.shape
    pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_args)
    target_mask = find_xrd_peaks_batch(target_pattern, **peak_args)
    pred_indices = _get_batch_peak_indices(pred_mask)
    target_indices = _get_batch_peak_indices(target_mask)
    if pred_indices.shape[1] == 0 or target_indices.shape[1] == 0:
        return 0.0

    min_2theta, max_2theta = two_theta_range
    pred_2theta = min_2theta + (max_2theta - min_2theta) * pred_indices.float() / (length - 1)
    target_2theta = min_2theta + (max_2theta - min_2theta) * target_indices.float() / (length - 1)
    dists = torch.abs(pred_2theta.unsqueeze(2) - target_2theta.unsqueeze(1))
    valid_pairs = (pred_indices != -1).unsqueeze(2) & (target_indices != -1).unsqueeze(1)
    tol_val = (max_2theta - min_2theta) * tolerance / (length - 1)
    dists = dists.masked_fill(~(valid_pairs & (dists <= tol_val)), float("inf"))
    min_dists, _ = dists.min(dim=2)
    matched = min_dists != float("inf")
    if not matched.any():
        return 0.0

    return (min_dists[matched].sum() / matched.sum()).item()


def calculate_fwhm_error(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    window_size: int = 50,
    **peak_args: Any,
) -> float:
    """Calculate FWHM error."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    _, length = pred_pattern.shape
    device = pred_pattern.device
    pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_args)
    target_mask = find_xrd_peaks_batch(target_pattern, **peak_args)
    pred_indices = _get_batch_peak_indices(pred_mask)
    target_indices = _get_batch_peak_indices(target_mask)
    if pred_indices.shape[1] == 0 or target_indices.shape[1] == 0:
        return 0.0

    dists = torch.abs(pred_indices.unsqueeze(2).float() - target_indices.unsqueeze(1).float())
    valid = (pred_indices != -1).unsqueeze(2) & (target_indices != -1).unsqueeze(1) & (dists <= 10)
    min_dists, min_target_rel = dists.masked_fill(~valid, float("inf")).min(dim=2)
    matched = min_dists != float("inf")
    if not matched.any():
        return 0.0

    batch_idx, pred_rel_idx = torch.nonzero(matched, as_tuple=True)
    pred_peaks = pred_indices[batch_idx, pred_rel_idx]
    target_peaks = target_indices[batch_idx, min_target_rel[batch_idx, pred_rel_idx]]

    def compute_fwhm(patterns: torch.Tensor, peak_idx: torch.Tensor) -> torch.Tensor:
        count = peak_idx.shape[0]
        half_window = window_size // 2
        offsets = torch.arange(-half_window, half_window + 1, device=device)
        window_indices = torch.clamp(peak_idx.unsqueeze(1) + offsets.unsqueeze(0), 0, length - 1)
        windows = torch.gather(patterns, 1, window_indices)
        half_max = patterns[torch.arange(count, device=device), peak_idx].unsqueeze(1) / 2.0

        left = torch.flip(windows[:, :half_window], [1]) <= half_max
        left_dist = torch.where(
            left.any(dim=1),
            torch.argmax(left.float(), dim=1) + 1,
            torch.tensor(half_window, device=device),
        )

        right = windows[:, half_window + 1:] <= half_max
        right_dist = torch.where(
            right.any(dim=1),
            torch.argmax(right.float(), dim=1) + 1,
            torch.tensor(right.shape[1], device=device),
        )
        return (left_dist + right_dist).float()

    pred_fwhm = compute_fwhm(pred_pattern[batch_idx], pred_peaks)
    target_fwhm = compute_fwhm(target_pattern[batch_idx], target_peaks)
    return torch.abs(pred_fwhm - target_fwhm).mean().item()


def calculate_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity."""
    vec1_norm = F.normalize(vec1, p=2, dim=1)
    vec2_norm = F.normalize(vec2, p=2, dim=1)
    return torch.mm(vec1_norm, vec2_norm.t())


def calculate_retrieval_topk(
    separated_patterns: torch.Tensor,
    true_phase_ids: torch.Tensor,
    active_mask: torch.Tensor,
    reference_library: torch.Tensor,
    reference_ids: List[int],
    top_ks: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Calculate id accuracy at top-k."""
    top_ks = top_ks or list(range(1, 11))
    hits = {f"id_acc_top{k}": 0.0 for k in top_ks}

    active_patterns = separated_patterns[active_mask]
    active_ids = true_phase_ids[active_mask].cpu().numpy()
    if active_patterns.shape[0] == 0:
        return hits

    ref_id_to_idx = {rid: i for i, rid in enumerate(reference_ids)}
    similarities = calculate_cosine_similarity(active_patterns, reference_library)
    max_k = min(max(top_ks), similarities.shape[1])
    _, top_indices = torch.topk(similarities, k=max_k, dim=1)
    top_indices = top_indices.cpu().numpy()

    total = 0
    for row_idx, true_id in enumerate(active_ids):
        true_idx = ref_id_to_idx.get(int(true_id))
        if true_idx is None:
            continue
        total += 1
        for k in top_ks:
            if true_idx in top_indices[row_idx, : min(k, max_k)]:
                hits[f"id_acc_top{k}"] += 1.0

    if total == 0:
        return hits
    return {key: value / total for key, value in hits.items()}


def calculate_all_metrics(
    pred_patterns: torch.Tensor,
    target_patterns: torch.Tensor,
    phase_ids: torch.Tensor,
    reference_library: Optional[torch.Tensor] = None,
    reference_ids: Optional[List[int]] = None,
    active_threshold: float = 1e-4,
    two_theta_range: Tuple[float, float] = (10.0, 80.0),
) -> Dict[str, float]:
    """Calculate the baseline evaluation metrics."""
    from scipy.optimize import linear_sum_assignment

    batch_size, _, _ = pred_patterns.shape
    device = pred_patterns.device
    target_is_active = torch.sum(target_patterns ** 2, dim=-1) > active_threshold
    aligned_preds = torch.zeros_like(target_patterns)
    metric_sums = {
        "pearson_corr": 0.0,
        "si_sdr": 0.0,
        "sir": 0.0,
        "sar": 0.0,
        "delta_2theta": 0.0,
        "fwhm_error": 0.0,
    }
    active_pairs = 0

    for batch_idx in range(batch_size):
        active_target_mask = target_is_active[batch_idx]
        if active_target_mask.sum().item() == 0:
            continue

        active_target_patterns = target_patterns[batch_idx, active_target_mask]
        cost_matrix = torch.cdist(pred_patterns[batch_idx], active_target_patterns, p=2) ** 2
        pred_idx, target_rel_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        target_abs_idx = torch.where(active_target_mask)[0][torch.as_tensor(target_rel_idx, device=device)]
        pred_idx = torch.as_tensor(pred_idx, device=device)

        aligned_preds[batch_idx, target_abs_idx] = pred_patterns[batch_idx, pred_idx]
        matched_pred = pred_patterns[batch_idx, pred_idx]
        matched_target = target_patterns[batch_idx, target_abs_idx]
        target_sum = matched_target.sum(dim=0)

        for pair_idx in range(matched_pred.shape[0]):
            pred_pattern = matched_pred[pair_idx]
            target_pattern = matched_target[pair_idx]
            interference = target_sum - target_pattern
            sir, sar = calculate_sir_sar(pred_pattern, target_pattern, interference=interference)

            metric_sums["pearson_corr"] += calculate_pearson_correlation(pred_pattern, target_pattern)
            metric_sums["si_sdr"] += calculate_sisdr(pred_pattern, target_pattern)
            metric_sums["sir"] += sir
            metric_sums["sar"] += sar
            metric_sums["delta_2theta"] += calculate_peak_shift_delta_2theta(
                pred_pattern,
                target_pattern,
                two_theta_range=two_theta_range,
            )
            metric_sums["fwhm_error"] += calculate_fwhm_error(pred_pattern, target_pattern)
            active_pairs += 1

    results = {key: 0.0 for key in metric_sums}
    if active_pairs > 0:
        results = {key: value / active_pairs for key, value in metric_sums.items()}

    id_metrics = {f"id_acc_top{k}": 0.0 for k in range(1, 11)}
    if reference_library is not None and reference_ids is not None:
        id_metrics = calculate_retrieval_topk(
            aligned_preds,
            phase_ids,
            target_is_active,
            reference_library,
            reference_ids,
        )
    results.update(id_metrics)
    return results


class SeparationLoss(nn.Module):
    """Permutation-invariant MSE loss."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, num_sources, length = target.shape
        device = target.device

        perms = list(itertools.permutations(range(num_sources)))
        perms_tensor = torch.tensor(perms, device=device)
        num_perms = len(perms)

        target_expanded = target.unsqueeze(1).expand(-1, num_perms, -1, -1)
        gather_idx = perms_tensor.view(1, num_perms, num_sources, 1).expand(batch_size, -1, -1, length)
        target_perms = torch.gather(target_expanded, 2, gather_idx)

        mse_all = torch.mean((pred.unsqueeze(1) - target_perms) ** 2, dim=(2, 3))
        min_mse, _ = torch.min(mse_all, dim=1)
        return torch.mean(min_mse)
