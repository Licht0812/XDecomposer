"""
XRD Evaluation Metrics Module (Comprehensive Version).
Adapted from user reference for DARA integration.
"""

import warnings
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# =============================================================================
# 1. Separation Quality Metrics (Waveform-based)
# =============================================================================

def calculate_rwp(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Calculate R-weighted Profile (Rwp)."""
    if pred_pattern.dim() == 1:
        pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    target_pattern, pred_pattern = torch.clamp(target_pattern, min=0), torch.clamp(pred_pattern, min=0)
    diff_sq = (target_pattern - pred_pattern) ** 2
    numerator = torch.sum(diff_sq, dim=-1)
    denominator = torch.sum(target_pattern ** 2, dim=-1) + epsilon
    return torch.sqrt(numerator / denominator).mean().item()

def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    """Calculate Pearson correlation coefficient."""
    if pred_pattern.dim() == 1:
        pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    pred_mean, target_mean = pred_pattern.mean(dim=-1, keepdim=True), target_pattern.mean(dim=-1, keepdim=True)
    pred_centered, target_centered = pred_pattern - pred_mean, target_pattern - target_mean
    numerator = (pred_centered * target_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)
    return (numerator / (pred_std * target_std + 1e-8)).mean().item()

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if pred.ndim > 2:
        L = pred.shape[-1]
        pred, target = pred.reshape(-1, L), target.reshape(-1, L)
    elif pred.ndim == 1:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    dot_product = torch.sum(pred * target, dim=-1)
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    alpha = dot_product / target_energy
    e_target = alpha.unsqueeze(-1) * target
    e_res = pred - e_target
    signal_energy, noise_energy = torch.sum(e_target ** 2, dim=-1), torch.sum(e_res ** 2, dim=-1) + eps
    ratio = torch.clamp(signal_energy / noise_energy, min=1e-10)
    return (10 * torch.log10(ratio)).mean().item()

def calculate_sir_sar(pred: torch.Tensor, target: torch.Tensor, interference: Optional[torch.Tensor] = None, eps: float = 1e-8) -> Tuple[float, float]:
    """Calculate SIR (Signal-to-Interference Ratio) and SAR (Signal-to-Artifacts Ratio)."""
    if pred.ndim == 1:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    artifacts = pred - target
    target_energy, artifacts_energy = torch.sum(target ** 2, dim=-1) + eps, torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)
    interference = interference if interference is not None else artifacts
    if interference.ndim == 1: interference = interference.unsqueeze(0)
    sir = 10 * torch.log10(target_energy / (torch.sum(interference ** 2, dim=-1) + eps))
    return sir.mean().item(), sar.mean().item()

# =============================================================================
# 2. Peak-based Quality Metrics
# =============================================================================

def find_xrd_peaks_batch(xrd_patterns: torch.Tensor, height_threshold: float = 0.05, distance: int = 10) -> torch.Tensor:
    """Detects peaks in a BATCH of XRD patterns."""
    if xrd_patterns.dim() == 1: xrd_patterns = xrd_patterns.unsqueeze(0)
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
    B, L = mask.shape
    device = mask.device
    idx_map = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    masked_idxs = torch.where(mask, idx_map, torch.tensor(L, device=device))
    sorted_idxs, _ = torch.sort(masked_idxs, dim=1)
    max_peaks = mask.sum(dim=1).max().item()
    if max_peaks == 0: return torch.full((B, 0), -1, device=device, dtype=torch.long)
    dense_idxs = sorted_idxs[:, :max_peaks].clone()
    dense_idxs[dense_idxs == L] = -1
    return dense_idxs

def calculate_peak_shift_delta_2theta(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, two_theta_range: Tuple[float, float] = (5.0, 90.0), tolerance: int = 10, **peak_args: Any) -> float:
    """Calculate mean peak position shift in Δ2θ units."""
    if pred_pattern.dim() == 1: pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    B, L = pred_pattern.shape
    device = pred_pattern.device
    p_mask, t_mask = find_xrd_peaks_batch(pred_pattern, **peak_args), find_xrd_peaks_batch(target_pattern, **peak_args)
    p_idxs, t_idxs = _get_batch_peak_indices(p_mask), _get_batch_peak_indices(t_mask)
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return 0.0
    min_2t, max_2t = two_theta_range
    p_2theta = min_2t + (max_2t - min_2t) * p_idxs.float() / (L - 1)
    t_2theta = min_2t + (max_2t - min_2t) * t_idxs.float() / (L - 1)
    dists = torch.abs(p_2theta.unsqueeze(2) - t_2theta.unsqueeze(1))
    valid_pair_mask = (p_idxs != -1).unsqueeze(2) & (t_idxs != -1).unsqueeze(1)
    tol_val = (max_2t - min_2t) * tolerance / (L - 1)
    final_mask = valid_pair_mask & (dists <= tol_val)
    dists_inf = dists.masked_fill(~final_mask, float('inf'))
    min_dists, _ = dists_inf.min(dim=2)
    matched_mask = min_dists != float('inf')
    if not matched_mask.any(): return 0.0
    return (min_dists[matched_mask].sum() / matched_mask.sum()).item()

def calculate_fwhm_error(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, window_size: int = 50, **peak_args: Any) -> float:
    """Calculate FWHM error (Vectorized)."""
    if pred_pattern.dim() == 1: pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    B, L = pred_pattern.shape
    device = pred_pattern.device
    p_mask, t_mask = find_xrd_peaks_batch(pred_pattern, **peak_args), find_xrd_peaks_batch(target_pattern, **peak_args)
    p_idxs, t_idxs = _get_batch_peak_indices(p_mask), _get_batch_peak_indices(t_mask)
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return 0.0
    dists = torch.abs(p_idxs.unsqueeze(2).float() - t_idxs.unsqueeze(1).float())
    final_mask = (p_idxs != -1).unsqueeze(2) & (t_idxs != -1).unsqueeze(1) & (dists <= 10)
    min_dists, min_t_indices_rel = dists.masked_fill(~final_mask, float('inf')).min(dim=2)
    matched_mask = min_dists != float('inf')
    if not matched_mask.any(): return 0.0
    b_indices, p_rel_indices = torch.nonzero(matched_mask, as_tuple=True)
    p_peaks_l, t_peaks_l = p_idxs[b_indices, p_rel_indices], t_idxs[b_indices, min_t_indices_rel[b_indices, p_rel_indices]]
    def compute_fwhm(patterns, peak_indices):
        N = peak_indices.shape[0]
        w_half = window_size // 2
        window_indices = torch.clamp(peak_indices.unsqueeze(1) + torch.arange(-w_half, w_half + 1, device=device), 0, L - 1)
        windows = torch.gather(patterns, 1, window_indices)
        half_max = patterns[torch.arange(N), peak_indices].unsqueeze(1) / 2.0
        left_rev = torch.flip(windows[:, :w_half], [1])
        dist_l = torch.where(left_rev <= half_max, torch.argmax((left_rev <= half_max).float(), 1) + 1, torch.tensor(w_half, device=device))
        right_part = windows[:, w_half+1:]
        dist_r = torch.where(right_part <= half_max, torch.argmax((right_part <= half_max).float(), 1) + 1, torch.tensor(right_part.shape[1], device=device))
        return (dist_l + dist_r).float()
    return torch.abs(compute_fwhm(pred_pattern[b_indices], p_peaks_l) - compute_fwhm(target_pattern[b_indices], t_peaks_l)).mean().item()

def calculate_intensity_ratio_consistency(pred_patterns: torch.Tensor, target_patterns: torch.Tensor, tolerance: int = 10, **peak_args: Any) -> float:
    """Calculate intensity ratio consistency."""
    if pred_patterns.dim() == 1: pred_patterns, target_patterns = pred_patterns.unsqueeze(0), target_patterns.unsqueeze(0)
    p_mask, t_mask = find_xrd_peaks_batch(pred_patterns, **peak_args), find_xrd_peaks_batch(target_patterns, **peak_args)
    p_idxs, t_idxs = _get_batch_peak_indices(p_mask), _get_batch_peak_indices(t_mask)
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return 0.0
    dists = torch.abs(p_idxs.unsqueeze(2).float() - t_idxs.unsqueeze(1).float())
    final_mask = (p_idxs != -1).unsqueeze(2) & (t_idxs != -1).unsqueeze(1) & (dists <= tolerance)
    min_dists, min_t_indices_rel = dists.masked_fill(~final_mask, float('inf')).min(dim=2)
    matched_mask = min_dists != float('inf')
    safe_p_idxs = torch.where(p_idxs != -1, p_idxs, torch.zeros_like(p_idxs))
    p_intensities = torch.gather(pred_patterns, 1, safe_p_idxs).masked_fill(p_idxs == -1, 0.0)
    safe_min_t = torch.where(matched_mask, min_t_indices_rel, torch.zeros_like(min_t_indices_rel))
    matched_t_l_idxs = torch.gather(t_idxs, 1, safe_min_t)
    safe_matched_t_l = torch.where(matched_t_l_idxs != -1, matched_t_l_idxs, torch.zeros_like(matched_t_l_idxs))
    t_intensities = torch.gather(target_patterns, 1, safe_matched_t_l)
    p_max, t_max = torch.clamp(pred_patterns.max(1)[0].unsqueeze(1), min=1e-6), torch.clamp(target_patterns.max(1)[0].unsqueeze(1), min=1e-6)
    ratio_diff = torch.abs(p_intensities / p_max - t_intensities / t_max) * matched_mask.float()
    count_per_sample = matched_mask.float().sum(dim=1)
    has_matches = count_per_sample > 0
    if has_matches.sum() == 0: return 0.0
    return (1.0 - (ratio_diff.sum(1)[has_matches] / count_per_sample[has_matches])).mean().item()

# =============================================================================
# 3. Phase Identification & Ranking Metrics
# =============================================================================

def calculate_activity_metrics(act_logits: torch.Tensor, target_is_active: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate F1, Precision, Recall, and Exact Match for phase activity."""
    act_pred = (torch.sigmoid(act_logits) > threshold).float()
    tp = (act_pred * target_is_active).sum()
    fp = (act_pred * (1 - target_is_active)).sum()
    fn = ((1 - act_pred) * target_is_active).sum()
    
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    exact_match = (act_pred == target_is_active).all(dim=1).float().mean().item()
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'exact_match': exact_match}

def calculate_ranking_metrics(act_logits: torch.Tensor, best_perms: torch.Tensor, target_is_active: torch.Tensor) -> Dict[str, Any]:
    """Calculate Top-K Accuracy and Oracle Exact Match."""
    B, K = act_logits.shape
    _, sorted_indices = torch.sort(act_logits, dim=1, descending=True)
    # best_perms maps Pred Index -> Target Index
    matched_target_indices = torch.gather(best_perms, 1, sorted_indices)
    is_hit = torch.gather(target_is_active, 1, matched_target_indices) # [B, K]
    
    num_actives = target_is_active.sum(dim=1).long()
    oracle_hits = 0
    topk_acc = {f'top{k}_acc': 0.0 for k in [1, 10]} # Standard ones
    
    for i in range(B):
        M = num_actives[i].item()
        if M > 0:
            if is_hit[i, :M].sum().item() == M: oracle_hits += 1
            for k in [1, 10]:
                if k <= K:
                    if is_hit[i, :k].any(): topk_acc[f'top{k}_acc'] += 1.0
        else:
            oracle_hits += 1
            for k in [1, 10]: topk_acc[f'top{k}_acc'] += 1.0

    metrics = {f'top{k}_acc': v / B for k, v in topk_acc.items()}
    metrics['oracle_exact_match'] = oracle_hits / B
    return metrics

# =============================================================================
# 4. Quantitative Analysis Metrics
# =============================================================================

def calculate_quantitative_metrics(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> Dict[str, Any]:
    """Calculate Quantitative Mean Absolute Error (Quant MAE) in percentage points."""
    # pred/target: [B, K, L]
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    
    # MAE in percentage points (0-100)
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return {'quant_mae': mae, 'pred_pct': pred_pct, 'target_pct': target_pct}

# =============================================================================
# Global Wrapper
# =============================================================================

def calculate_id_accuracy(
    matched_ids_with_scores: List[List[Tuple[int, float]]], 
    ground_truth_ids: torch.Tensor,
    topk_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calculate Top-K ID accuracy for each active ground truth phase.
    matched_ids_with_scores: List (for each slot) of List of (id, score)
    ground_truth_ids: [K] tensor of GT crystal IDs
    """
    results = {f'id_acc_top{k}': 0.0 for k in topk_values}
    num_active = 0
    
    # ground_truth_ids: [K]
    for i, gt_id in enumerate(ground_truth_ids):
        gt_id_item = gt_id.item()
        if gt_id_item == -1: continue # Padding or inactive
        
        num_active += 1
        # matched_ids_with_scores[i] is the top-K list for this predicted slot
        if i < len(matched_ids_with_scores):
            pred_top_ids = [item[0] for item in matched_ids_with_scores[i]]
            for k in topk_values:
                if gt_id_item in pred_top_ids[:k]:
                    results[f'id_acc_top{k}'] += 1.0
                    
    if num_active > 0:
        for k in topk_values:
            results[f'id_acc_top{k}'] /= num_active
            
    return results

def calculate_all_metrics(
    aligned_preds: torch.Tensor, 
    targets: torch.Tensor,
    act_logits: Optional[torch.Tensor] = None,
    best_perms: Optional[torch.Tensor] = None,
    matched_ids_with_scores: Optional[List[List[Tuple[int, float]]]] = None,
    two_theta_range: Tuple[float, float] = (5.0, 90.0),
    **kwargs
) -> Dict[str, float]:
    """Utility to calculate all relevant metrics at once."""
    # 1. Separation Metrics
    is_active = torch.sum(targets ** 2, dim=-1) > 1e-6
    # For separation metrics, we only care about slots that are active in BOTH targets and aligned_preds
    # But often we just use is_active from targets to see how well we recovered those.
    
    # Handle batch dimension if present
    if aligned_preds.dim() == 3: # [B, K, L]
        # For simplicity, we assume batch size 1 for now or average across batch
        B = aligned_preds.shape[0]
        all_res = []
        for b in range(B):
            res = calculate_all_metrics(
                aligned_preds[b], 
                targets[b], 
                act_logits[b] if act_logits is not None else None,
                best_perms[b] if best_perms is not None else None,
                matched_ids_with_scores[b] if matched_ids_with_scores is not None else None,
                two_theta_range,
                **kwargs
            )
            all_res.append(res)
        
        # Average results
        avg_res = {}
        for k in all_res[0].keys():
            avg_res[k] = np.mean([r[k] for r in all_res])
        return avg_res

    # Single sample logic [K, L]
    p_active, t_active = aligned_preds[is_active], targets[is_active]
    
    # Extract gt_ids and peak detection args
    gt_ids = kwargs.get('gt_ids')
    peak_kwargs = {k: v for k, v in kwargs.items() if k != 'gt_ids'}
    
    results = {}
    if p_active.shape[0] > 0:
        results.update({
            'rwp': calculate_rwp(p_active, t_active),
            'pearson': calculate_pearson_correlation(p_active, t_active),
            'si_sdr': calculate_sisdr(p_active, t_active),
            'delta_2theta': calculate_peak_shift_delta_2theta(p_active, t_active, two_theta_range, **peak_kwargs),
            'fwhm_error': calculate_fwhm_error(p_active, t_active, **peak_kwargs),
            'intensity_consistency': calculate_intensity_ratio_consistency(p_active, t_active, **peak_kwargs)
        })
    else:
        results.update({'rwp': 1.0, 'pearson': 0.0, 'si_sdr': -20.0, 'delta_2theta': 0.0, 'fwhm_error': 0.0, 'intensity_consistency': 0.0})
    
    # 2. Activity Metrics
    target_is_active = is_active.float().unsqueeze(0) # [1, K]
    if act_logits is not None:
        act_logits_2d = act_logits.unsqueeze(0) # [1, K]
        act_res = calculate_activity_metrics(act_logits_2d, target_is_active)
        results.update({f'activity_{k}': v for k, v in act_res.items()})
        
        if best_perms is not None:
            best_perms_2d = best_perms.unsqueeze(0) # [1, K]
            rank_res = calculate_ranking_metrics(act_logits_2d, best_perms_2d, target_is_active)
            results.update(rank_res)
            
    # 3. ID Accuracy
    if matched_ids_with_scores is not None:
        # targets: [K, L], we need gt_ids from somewhere. 
        # In this context, we'll assume gt_ids are passed in kwargs or handled outside.
        if gt_ids is not None:
            id_res = calculate_id_accuracy(matched_ids_with_scores, gt_ids)
            results.update(id_res)
            
    # 4. Quantitative Metrics
    quant_res = calculate_quantitative_metrics(aligned_preds.unsqueeze(0), targets.unsqueeze(0))
    results['quant_mae'] = quant_res['quant_mae']
    
    return results
