"""
XRD 评估指标工具模块
只包含以下7个核心指标：
1. R-weighted Profile (Rwp)
2. Pearson 相关系数
3. SI-SDR
4. SIR / SAR (Signal-to-Interference Ratio / Signal-to-Artifacts Ratio)
5. 峰位偏移量 (Δ2θ)
6. 半高宽 (FWHM) 误差
7. 强度比例一致性
"""

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
    Detects peaks in a BATCH of XRD patterns using Pure PyTorch.
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
    
    # 3. Distance Suppression (MaxPool)
    if distance > 0:
        kernel_size = 2 * distance + 1
        x_pad_pool = F.pad(xrd_patterns.unsqueeze(1), (distance, distance), value=-1e9)
        max_in_window = F.max_pool1d(x_pad_pool, kernel_size=kernel_size, stride=1).squeeze(1)
        is_window_max = torch.abs(xrd_patterns - max_in_window) < 1e-6
        mask = mask & is_window_max
        
    return mask


def calculate_rwp(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, epsilon: float = 1e-8) -> float:
    """
    Calculate R-weighted Profile (Rwp) - 晶体学标准残差因子。
    
    Rwp = sqrt(Σ w_i (y_obs_i - y_calc_i)^2 / Σ w_i y_obs_i^2)
    其中权重 w_i 通常取 y_obs_i（观测强度），这里简化为 w_i = 1。
    
    Args:
        pred_pattern: [B, L] or [L]
        target_pattern: [B, L] or [L]
    Returns:
        Rwp (scalar)
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
    
    SIR: 衡量目标信号与干扰信号（其他相）的比值
    SAR: 衡量目标信号与伪影（预测误差）的比值
    
    Args:
        pred: [B, L] 预测的单个相
        target: [B, L] 目标单个相
        interference: [B, L] 其他相的干扰（可选，如果为None则从pred中估计）
        eps: 防止除零
    Returns:
        (SIR, SAR) in dB
    """
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # SAR: Signal-to-Artifacts Ratio
    # artifacts = pred - target (预测误差)
    artifacts = pred - target
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    artifacts_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)
    
    # SIR: Signal-to-Interference Ratio
    # 如果没有提供interference，我们假设interference = pred - target（即预测误差）
    # 在实际应用中，interference可能是其他相的预测结果
    if interference is None:
        interference = artifacts  # 简化：用artifacts作为interference
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
    
    # 1. Create index map
    idx_map = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    
    # 2. Mask out non-peaks with a large value (L)
    # We want valid indices (0..L-1) at the start
    masked_idxs = torch.where(mask, idx_map, torch.tensor(L, device=device))
    
    # 3. Sort ascending
    sorted_idxs, _ = torch.sort(masked_idxs, dim=1)
    
    # 4. Determine max peaks
    num_peaks_per_sample = mask.sum(dim=1)
    max_peaks = num_peaks_per_sample.max().item()
    
    if max_peaks == 0:
        return torch.full((B, 0), -1, device=device, dtype=torch.long)
        
    # 5. Slice
    dense_idxs = sorted_idxs[:, :max_peaks]
    
    # 6. Replace L with -1
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
    
    # Filter peak_args
    peak_kwargs = {k: v for k, v in peak_args.items() if k in ['height_threshold', 'distance']}
    if pred_mask is None:
        pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_kwargs)
    if target_mask is None:
        target_mask = find_xrd_peaks_batch(target_pattern, **peak_kwargs)
    
    # 1. Get Peak Indices [B, M]
    p_idxs = _get_batch_peak_indices(pred_mask)
    t_idxs = _get_batch_peak_indices(target_mask)
    
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0:
        return 0.0
        
    # 2. Convert to 2Theta
    min_2t, max_2t = two_theta_range
    
    def idx_to_2theta(idx_tensor):
        return min_2t + (max_2t - min_2t) * idx_tensor.float() / (L - 1)
        
    p_2theta = idx_to_2theta(p_idxs)
    t_2theta = idx_to_2theta(t_idxs)
    
    # 3. Compute Distance Matrix [B, Mp, Mt]
    # Handle padding (-1)
    p_valid = p_idxs != -1
    t_valid = t_idxs != -1
    
    # Distances
    # unsqueeze to broadcast: [B, Mp, 1] - [B, 1, Mt]
    dists = torch.abs(p_2theta.unsqueeze(2) - t_2theta.unsqueeze(1))
    
    # 4. Mask invalid pairs & Tolerance
    valid_pair_mask = p_valid.unsqueeze(2) & t_valid.unsqueeze(1)
    
    tol_val = (max_2t - min_2t) * tolerance / (L - 1)
    valid_tolerance = dists <= tol_val
    
    final_mask = valid_pair_mask & valid_tolerance
    
    # 5. Find closest target for each pred
    dists_inf = dists.masked_fill(~final_mask, float('inf'))
    min_dists, _ = dists_inf.min(dim=2) # [B, Mp]
    
    # 6. Average
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

    # 1. Match Peaks (Similar to above)
    p_idxs = _get_batch_peak_indices(pred_mask)
    t_idxs = _get_batch_peak_indices(target_mask)
    
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0:
        return 0.0
        
    p_valid = p_idxs != -1
    t_valid = t_idxs != -1
    
    # Use indices distance
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
        
    # Get actual matched indices
    # p_idxs contains the pred peak indices [B, Mp]
    # min_t_indices_rel contains the index into t_idxs [B, Mp]
    # We need to gather the actual t indices
    
    # 2. Flatten for FWHM computation
    # We only care about matched pairs
    # [N_matches]
    b_indices, p_rel_indices = torch.nonzero(matched_mask, as_tuple=True)
    
    # Get Pred Peak Indices (in L)
    p_peaks_l = p_idxs[b_indices, p_rel_indices]
    
    # Get Target Peak Indices (in L)
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
        # We need gather(1, window_indices)
        windows = torch.gather(patterns, 1, window_indices) # [N, W]
        
        # Peak heights (center of window approx)
        # Actually peak is at 'centers', which corresponds to index w_half in 'offsets'
        # But clamping might shift it if near edge. 
        # Assuming peaks not at exact edge.
        peak_heights = patterns[torch.arange(N, device=device), peak_indices]
        half_max = peak_heights / 2.0
        
        # Scan Left (from center w_half down to 0)
        # We look at windows[:, :w_half]
        left_part = windows[:, :w_half] # [N, w_half]
        # Reverse to scan from center outwards
        left_part_rev = torch.flip(left_part, [1]) 
        # Find first <= half_max
        left_cond = left_part_rev <= half_max.unsqueeze(1)
        # argmax gives first index where True
        # If none True, gives 0. 
        # We check if any is True.
        has_left = left_cond.any(dim=1)
        left_dist = torch.argmax(left_cond.float(), dim=1) # 0-based index in rev
        # Distance from center = left_dist + 1
        # If not found, default to edge (w_half)
        dist_l = torch.where(has_left, left_dist + 1, torch.tensor(w_half, device=device))
        
        # Scan Right (from center w_half+1 to end)
        right_part = windows[:, w_half+1:] # [N, W - w_half - 1]
        right_cond = right_part <= half_max.unsqueeze(1)
        has_right = right_cond.any(dim=1)
        right_dist = torch.argmax(right_cond.float(), dim=1)
        dist_r = torch.where(has_right, right_dist + 1, torch.tensor(right_part.shape[1], device=device))
        
        # Total FWHM = dist_l + dist_r (approx)
        # Wait, if center is peak, dist to left half-max + dist to right half-max
        return (dist_l + dist_r).float()
    
    # 3. Compute
    # Gather relevant patterns [N_matches, L]
    p_patterns_subset = pred_pattern[b_indices]
    t_patterns_subset = target_pattern[b_indices]
    
    p_fwhm = compute_fwhm_batch(p_patterns_subset, p_peaks_l, window_size)
    t_fwhm = compute_fwhm_batch(t_patterns_subset, t_peaks_l, window_size)
    
    error = torch.abs(p_fwhm - t_fwhm).mean().item()
    return error


def calculate_intensity_ratio_consistency(
    pred_patterns: torch.Tensor,
    target_patterns: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
    tolerance: int = 10,
    **peak_args: Any
) -> float:
    """
    Calculate intensity ratio consistency (Vectorized).
    """
    if pred_patterns.dim() == 1:
        pred_patterns = pred_patterns.unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0)
    
    B, L = pred_patterns.shape
    device = pred_patterns.device
    
    if pred_mask is None:
        pred_mask = find_xrd_peaks_batch(pred_patterns, **peak_args)
    if target_mask is None:
        target_mask = find_xrd_peaks_batch(target_patterns, **peak_args)
        
    # 1. Match Peaks (Reuse logic)
    p_idxs = _get_batch_peak_indices(pred_mask)
    t_idxs = _get_batch_peak_indices(target_mask)
    
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0:
        return 0.0
        
    p_valid = p_idxs != -1
    t_valid = t_idxs != -1
    dists = torch.abs(p_idxs.unsqueeze(2).float() - t_idxs.unsqueeze(1).float())
    
    valid_pair_mask = p_valid.unsqueeze(2) & t_valid.unsqueeze(1)
    valid_tolerance = dists <= tolerance
    final_mask = valid_pair_mask & valid_tolerance
    
    dists_inf = dists.masked_fill(~final_mask, float('inf'))
    min_dists, min_t_indices_rel = dists_inf.min(dim=2)
    
    matched_mask = min_dists != float('inf')
    
    # 2. Calculate Consistency per Sample
    # We need to group by sample to normalize by max intensity
    
    # Get Intensities [B, Mp]
    # Gather from pred_patterns [B, L] using p_idxs [B, Mp]
    # We need to handle -1 in p_idxs by clamping or masking
    # Use 0 for -1, then mask result
    safe_p_idxs = p_idxs.clone()
    safe_p_idxs[p_idxs == -1] = 0
    p_intensities = torch.gather(pred_patterns, 1, safe_p_idxs) # [B, Mp]
    p_intensities = p_intensities.masked_fill(~p_valid, 0.0)
    
    # Get Target Intensities [B, Mp] (Aligned to Preds)
    # We need matched target indices.
    # min_t_indices_rel is [B, Mp] index into t_idxs
    safe_min_t = min_t_indices_rel.clone()
    # Where no match, just point to 0 (will be masked out)
    safe_min_t[~matched_mask] = 0 
    
    # Gather t_idxs from safe_min_t
    matched_t_l_idxs = torch.gather(t_idxs, 1, safe_min_t) # [B, Mp]
    
    # Gather intensities
    safe_matched_t_l = matched_t_l_idxs.clone()
    safe_matched_t_l[matched_t_l_idxs == -1] = 0
    t_intensities = torch.gather(target_patterns, 1, safe_matched_t_l)
    
    # Mask out unmatched or invalid
    final_valid = matched_mask & p_valid # p_valid is redundant if matched_mask is derived correctly
    
    p_matched_int = p_intensities
    t_matched_int = t_intensities
    
    # 3. Max Intensity per sample (from ALL peaks or matched?)
    # Usually relative to max peak in the pattern
    # pred_patterns.max(dim=1)
    p_max = pred_patterns.max(dim=1)[0].unsqueeze(1) # [B, 1]
    t_max = target_patterns.max(dim=1)[0].unsqueeze(1)
    
    # Avoid div by zero
    p_max = torch.clamp(p_max, min=1e-6)
    t_max = torch.clamp(t_max, min=1e-6)
    
    p_ratios = p_matched_int / p_max
    t_ratios = t_matched_int / t_max
    
    # 4. Error
    ratio_diff = torch.abs(p_ratios - t_ratios)
    
    # Average over matched peaks per sample?
    # Or average over all matched peaks in batch?
    # Logic in original: "consistency = 1 - mean(abs(diff))" per sample.
    
    # We sum errors where valid
    ratio_diff_valid = ratio_diff * final_valid.float()
    sum_diff_per_sample = ratio_diff_valid.sum(dim=1)
    count_per_sample = final_valid.float().sum(dim=1)
    
    # Handle samples with matches
    has_matches = count_per_sample > 0
    
    mean_diff_per_sample = torch.zeros_like(sum_diff_per_sample)
    mean_diff_per_sample[has_matches] = sum_diff_per_sample[has_matches] / count_per_sample[has_matches]
    
    consistency_per_sample = 1.0 - mean_diff_per_sample
    
    # Only average over samples that had matches?
    # Original: consistency_sum += consistency; count += 1 (if matches found)
    
    if has_matches.sum() == 0:
        return 0.0
        
    final_consistency = consistency_per_sample[has_matches].mean().item()
    return final_consistency


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
        calc_detailed: Whether to calculate slow peak-based metrics (Delta 2theta, FWHM, Intensity)
        **kwargs: Additional arguments for peak detection
    Returns:
        Dictionary with metrics.
    """
    # Flatten if needed
    original_shape = pred_patterns.shape
    if pred_patterns.ndim == 3:
        B, K, L = pred_patterns.shape
        pred_patterns = pred_patterns.reshape(B * K, L)
        target_patterns = target_patterns.reshape(B * K, L)
    elif pred_patterns.ndim == 1:
        pred_patterns = pred_patterns.unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0)
    
    # Filter out silent targets (padding)
    # Calculate energy or max value to detect silence
    target_energy = torch.sum(target_patterns ** 2, dim=-1)
    is_active = target_energy > 1e-6
    
    if is_active.sum() == 0:
        # If no active targets in this batch, return zeros (or nan/inf handled gracefully)
        return {
            'rwp': 0.0, 'pearson_corr': 0.0, 'si_sdr': 0.0,
            'sir': 0.0, 'sar': 0.0, 
            'delta_2theta': 0.0, 'fwhm_error': 0.0, 'intensity_ratio_consistency': 0.0
        }
        
    pred_active = pred_patterns[is_active]
    target_active = target_patterns[is_active]
    
    metrics = {}
    
    # 1. R-weighted Profile
    metrics['rwp'] = calculate_rwp(pred_active, target_active)
    
    # 2. Pearson Correlation
    metrics['pearson_corr'] = calculate_pearson_correlation(pred_active, target_active)
    
    # 3. SI-SDR
    metrics['si_sdr'] = calculate_sisdr(pred_active, target_active)
    
    # 4. SIR / SAR
    sir, sar = calculate_sir_sar(pred_active, target_active)
    metrics['sir'] = sir
    metrics['sar'] = sar
    
    if calc_detailed:
        # 5. Peak Shift (Δ2θ)
        metrics['delta_2theta'] = calculate_peak_shift_delta_2theta(
            pred_active, target_active, two_theta_range=two_theta_range, **kwargs
        )
        
        # 6. FWHM Error
        metrics['fwhm_error'] = calculate_fwhm_error(pred_active, target_active, **kwargs)
        
        # 7. Intensity Ratio Consistency
        metrics['intensity_ratio_consistency'] = calculate_intensity_ratio_consistency(
            pred_active, target_active, **kwargs
        )
    else:
        metrics['delta_2theta'] = 0.0
        metrics['fwhm_error'] = 0.0
        metrics['intensity_ratio_consistency'] = 0.0
    
    return metrics

def calculate_peak_position_metrics_batch(preds, targets, threshold=0.1):
    return {
        'peak_recall': 0.0,
        'peak_precision': 0.0,
        'peak_f1': 0.0,
        'peak_mean_shift': 0.0
    }
