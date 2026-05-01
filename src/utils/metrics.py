import warnings
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def find_xrd_peaks_batch(
    xrd_patterns: torch.Tensor,
    height_threshold: float = 0.05,
    distance: int = 10
) -> torch.Tensor:
    """
    Detects peaks in a BATCH of XRD patterns.
    Returns: (B, L) Boolean tensor.
    """
    if xrd_patterns.dim() == 1:
        xrd_patterns = xrd_patterns.unsqueeze(0)

    # 1. Basic Threshold
    mask = xrd_patterns > height_threshold

    # 2. Local Maxima Check
    x_pad = F.pad(xrd_patterns, (1, 1), value=-1e9)
    is_local_max = (xrd_patterns > x_pad[:, :-2]) & (xrd_patterns > x_pad[:, 2:])
    mask = mask & is_local_max

    # 3. Distance Suppression
    if distance > 0:
        kernel_size = 2 * distance + 1
        x_pad_pool = F.pad(xrd_patterns.unsqueeze(1), (distance, distance), value=-1e9)
        max_in_window = F.max_pool1d(x_pad_pool, kernel_size=kernel_size, stride=1).squeeze(1)
        is_window_max = torch.abs(xrd_patterns - max_in_window) < 1e-6
        mask = mask & is_window_max

    return mask

def calculate_rwp(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Calculate R-weighted Profile (Rwp).
    """
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    target_pattern = torch.clamp(target_pattern, min=0)
    pred_pattern = torch.clamp(pred_pattern, min=0)

    diff_sq = (target_pattern - pred_pattern) ** 2
    numerator = torch.sum(diff_sq, dim=-1)
    denominator = torch.sum(target_pattern ** 2, dim=-1) + epsilon
    rwp = torch.sqrt(numerator / denominator)

    return rwp.mean().item()

def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    """
    Calculate Pearson correlation coefficient.

    r = Σ(x - x̄)(y - ȳ) / sqrt(Σ(x - x̄)^2 * Σ(y - ȳ)^2)

    Args:
        pred_pattern: [B, L] or [L]
        target_pattern: [B, L] or [L]
    Returns:
        Pearson correlation (scalar, range [-1, 1])
    """
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    # Center the data
    pred_mean = pred_pattern.mean(dim=-1, keepdim=True)
    target_mean = target_pattern.mean(dim=-1, keepdim=True)

    pred_centered = pred_pattern - pred_mean
    target_centered = target_pattern - target_mean

    # Calculate correlation
    numerator = (pred_centered * target_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)

    correlation = numerator / (pred_std * target_std + 1e-8)

    return correlation.mean().item()

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    SI-SDR = 10 * log10(||α * target||^2 / ||pred - α * target||^2)
    where α = <pred, target> / ||target||^2

    Args:
        pred: [B, L] or [L]
        target: [B, L] or [L]
    Returns:
        SI-SDR in dB (scalar)
    """
    if pred.ndim > 2:
        L = pred.shape[-1]
        pred = pred.reshape(-1, L)
        target = target.reshape(-1, L)
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

    # Avoid log10(0) -> -inf
    ratio = signal_energy / noise_energy
    ratio = torch.clamp(ratio, min=1e-10) # Clamp to -100 dB

    sisdr = 10 * torch.log10(ratio)

    return sisdr.mean().item()

def calculate_sir_sar(pred: torch.Tensor, target: torch.Tensor, interference: Optional[torch.Tensor] = None, eps: float = 1e-8) -> Tuple[float, float]:
    """
    Calculate SIR (Signal-to-Interference Ratio) and SAR (Signal-to-Artifacts Ratio).
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # SAR: Signal-to-Artifacts Ratio
    artifacts = pred - target
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    artifacts_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)

    # SIR: Signal-to-Interference Ratio
    if interference is None:
        interference = artifacts
    else:
        if interference.ndim == 1:
            interference = interference.unsqueeze(0)

    interference_energy = torch.sum(interference ** 2, dim=-1) + eps
    sir = 10 * torch.log10(target_energy / interference_energy)

    return sir.mean().item(), sar.mean().item()

def _get_batch_peak_indices(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, L] bool mask to [B, M] indices, padded with -1.
    """
    B, L = mask.shape
    device = mask.device

    idx_map = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    masked_idxs = torch.where(mask, idx_map, torch.tensor(L, device=device))

    sorted_idxs, _ = torch.sort(masked_idxs, dim=1)

    num_peaks_per_sample = mask.sum(dim=1)
    max_peaks = num_peaks_per_sample.max().item()

    if max_peaks == 0:
        return torch.full((B, 0), -1, device=device, dtype=torch.long)

    dense_idxs = sorted_idxs[:, :max_peaks]

    dense_idxs = dense_idxs.clone() # avoid view issues
    dense_idxs[dense_idxs == L] = -1

    return dense_idxs

def calculate_peak_shift_delta_2theta(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    two_theta_range: Tuple[float, float] = (5.0, 90.0),
    tolerance: int = 10,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    **peak_args: Any
) -> float:
    """
    Calculate mean peak position shift in Δ2θ units (Vectorized).
    """
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    B, L = pred_pattern.shape
    device = pred_pattern.device

    peak_kwargs = {k: v for k, v in peak_args.items() if k in ['height_threshold', 'distance']}
    if pred_mask is None:
        pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_kwargs)
    if target_mask is None:
        target_mask = find_xrd_peaks_batch(target_pattern, **peak_kwargs)

    p_idxs = _get_batch_peak_indices(pred_mask)
    t_idxs = _get_batch_peak_indices(target_mask)

    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0:
        return 0.0

    min_2t, max_2t = two_theta_range

    def idx_to_2theta(idx_tensor):
        return min_2t + (max_2t - min_2t) * idx_tensor.float() / (L - 1)

    p_2theta = idx_to_2theta(p_idxs)
    t_2theta = idx_to_2theta(t_idxs)

    p_valid = p_idxs != -1
    t_valid = t_idxs != -1

    dists = torch.abs(p_2theta.unsqueeze(2) - t_2theta.unsqueeze(1))

    valid_pair_mask = p_valid.unsqueeze(2) & t_valid.unsqueeze(1)

    tol_val = (max_2t - min_2t) * tolerance / (L - 1)
    valid_tolerance = dists <= tol_val

    final_mask = valid_pair_mask & valid_tolerance

    dists_inf = dists.masked_fill(~final_mask, float('inf'))
    min_dists, _ = dists_inf.min(dim=2) # [B, Mp]

    matched_mask = min_dists != float('inf')

    if not matched_mask.any():
        return 0.0

    total_shift = min_dists[matched_mask].sum()
    count = matched_mask.sum()

    return (total_shift / count).item()

def calculate_fwhm_error(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    window_size: int = 50,
    **peak_args: Any
) -> float:
    """
    Calculate FWHM error (Vectorized).
    """
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    B, L = pred_pattern.shape
    device = pred_pattern.device

    peak_kwargs = {k: v for k, v in peak_args.items() if k in ['height_threshold', 'distance']}
    if pred_mask is None:
        pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_kwargs)
    if target_mask is None:
        target_mask = find_xrd_peaks_batch(target_pattern, **peak_kwargs)

    p_idxs = _get_batch_peak_indices(pred_mask)
    t_idxs = _get_batch_peak_indices(target_mask)

    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0:
        return 0.0

    p_valid = p_idxs != -1
    t_valid = t_idxs != -1

    dists = torch.abs(p_idxs.unsqueeze(2).float() - t_idxs.unsqueeze(1).float())
    tolerance = 10 # index units

    valid_pair_mask = p_valid.unsqueeze(2) & t_valid.unsqueeze(1)
    valid_tolerance = dists <= tolerance
    final_mask = valid_pair_mask & valid_tolerance

    dists_inf = dists.masked_fill(~final_mask, float('inf'))
    min_dists, min_t_indices_rel = dists_inf.min(dim=2) # [B, Mp]

    matched_mask = min_dists != float('inf')
    if not matched_mask.any():
        return 0.0

    b_indices, p_rel_indices = torch.nonzero(matched_mask, as_tuple=True)

    p_peaks_l = p_idxs[b_indices, p_rel_indices]

    t_rel_indices = min_t_indices_rel[b_indices, p_rel_indices]
    t_peaks_l = t_idxs[b_indices, t_rel_indices]

    def compute_fwhm_batch(patterns, peak_indices, w_size):
        """
        patterns: [N, L] (extracted from batch)
        peak_indices: [N]
        """
        N = peak_indices.shape[0]
        # Extract windows
        # Create index range [N, W]
        w_half = w_size // 2

        # [N, 1]
        centers = peak_indices.unsqueeze(1)
        # [1, W]
        offsets = torch.arange(-w_half, w_half + 1, device=device).unsqueeze(0)
        # [N, W]
        window_indices = centers + offsets
        window_indices = torch.clamp(window_indices, 0, L - 1)

        # Gather values: patterns is [N, L]
        windows = torch.gather(patterns, 1, window_indices) # [N, W]

        # Peak heights (center of window approx)
        peak_heights = patterns[torch.arange(N, device=device), peak_indices]
        half_max = peak_heights / 2.0

        # Scan Left (from center w_half down to 0)
        left_part = windows[:, :w_half] # [N, w_half]
        left_part_rev = torch.flip(left_part, [1])
        left_cond = left_part_rev <= half_max.unsqueeze(1)
        has_left = left_cond.any(dim=1)
        left_dist = torch.argmax(left_cond.float(), dim=1)
        dist_l = torch.where(has_left, left_dist + 1, torch.tensor(w_half, device=device))

        # Scan Right (from center w_half+1 to end)
        right_part = windows[:, w_half+1:] # [N, W - w_half - 1]
        right_cond = right_part <= half_max.unsqueeze(1)
        has_right = right_cond.any(dim=1)
        right_dist = torch.argmax(right_cond.float(), dim=1)
        dist_r = torch.where(has_right, right_dist + 1, torch.tensor(right_part.shape[1], device=device))

        # Total FWHM = dist_l + dist_r (approx)
        return (dist_l + dist_r).float()

    p_patterns_subset = pred_pattern[b_indices]
    t_patterns_subset = target_pattern[b_indices]

    p_fwhm = compute_fwhm_batch(p_patterns_subset, p_peaks_l, window_size)
    t_fwhm = compute_fwhm_batch(t_patterns_subset, t_peaks_l, window_size)

    error = torch.abs(p_fwhm - t_fwhm).mean().item()
    return error

def calculate_separation_metrics(
    pred_patterns: torch.Tensor,
    target_patterns: torch.Tensor,
    two_theta_range: Tuple[float, float] = (5.0, 90.0),
    calc_detailed: bool = True,
    **kwargs: Any
) -> Dict[str, float]:
    """
    Calculate separation metrics for XRD patterns.

    Args:
        pred_patterns: [B, K, L] or [B, L] or [L]
        target_patterns: [B, K, L] or [B, L] or [L]
        two_theta_range: (min_2theta, max_2theta) for Δ2θ calculation
        **kwargs: Additional arguments for peak detection
    Returns:
        Dictionary with metrics.
    """
    if pred_patterns.ndim == 1:
        pred_patterns = pred_patterns.unsqueeze(0).unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0).unsqueeze(0)
    elif pred_patterns.ndim == 2:
        pred_patterns = pred_patterns.unsqueeze(1)
        target_patterns = target_patterns.unsqueeze(1)
    elif pred_patterns.ndim != 3:
        raise ValueError("pred_patterns must be 1D, 2D, or 3D")

    target_energy = torch.sum(target_patterns ** 2, dim=-1)
    is_active = target_energy > 1e-6

    if is_active.sum().item() == 0:
        return {
            'rwp': 0.0, 'pearson_corr': 0.0, 'si_sdr': 0.0,
            'sir': 0.0, 'sar': 0.0,
            'delta_2theta': 0.0, 'fwhm_error': 0.0
        }

    metric_sums = {
        'rwp': 0.0,
        'pearson_corr': 0.0,
        'si_sdr': 0.0,
        'sir': 0.0,
        'sar': 0.0,
        'delta_2theta': 0.0,
        'fwhm_error': 0.0,
    }
    active_count = 0

    for batch_idx in range(pred_patterns.shape[0]):
        active_indices = torch.where(is_active[batch_idx])[0]
        if active_indices.numel() == 0:
            continue

        pred_active = pred_patterns[batch_idx, active_indices]
        target_active = target_patterns[batch_idx, active_indices]
        target_sum = target_active.sum(dim=0)

        for pair_idx in range(pred_active.shape[0]):
            pred_pattern = pred_active[pair_idx]
            target_pattern = target_active[pair_idx]
            interference = None
            if target_active.shape[0] > 1:
                interference = target_sum - target_pattern

            sir, sar = calculate_sir_sar(pred_pattern, target_pattern, interference=interference)
            metric_sums['rwp'] += calculate_rwp(pred_pattern, target_pattern)
            metric_sums['pearson_corr'] += calculate_pearson_correlation(pred_pattern, target_pattern)
            metric_sums['si_sdr'] += calculate_sisdr(pred_pattern, target_pattern)
            metric_sums['sir'] += sir
            metric_sums['sar'] += sar
            if calc_detailed:
                metric_sums['delta_2theta'] += calculate_peak_shift_delta_2theta(
                    pred_pattern,
                    target_pattern,
                    two_theta_range=two_theta_range,
                    **kwargs,
                )
                metric_sums['fwhm_error'] += calculate_fwhm_error(pred_pattern, target_pattern, **kwargs)
            active_count += 1

    if active_count == 0:
        return {key: 0.0 for key in metric_sums}

    metrics = {key: value / active_count for key, value in metric_sums.items()}
    if not calc_detailed:
        metrics['delta_2theta'] = 0.0
        metrics['fwhm_error'] = 0.0
    return metrics

def calculate_peak_position_metrics_batch(preds, targets, threshold=0.1):
    return {
        'peak_recall': 0.0,
        'peak_precision': 0.0,
        'peak_f1': 0.0,
        'peak_mean_shift': 0.0
    }
