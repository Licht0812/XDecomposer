import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import itertools

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
    # Ensure patterns are at least float32 for padding with large negative values
    original_dtype = xrd_patterns.dtype
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
        is_below_l = left_rev <= half_max
        dist_l = torch.where(is_below_l.any(dim=1), torch.argmax(is_below_l.float(), dim=1) + 1, torch.tensor(w_half, device=device))
        
        right_part = windows[:, w_half+1:]
        is_below_r = right_part <= half_max
        dist_r = torch.where(is_below_r.any(dim=1), torch.argmax(is_below_r.float(), dim=1) + 1, torch.tensor(right_part.shape[1], device=device))
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
# 3. Activity / Presence Metrics (Multi-label Binary Classification)
# =============================================================================

def calculate_activity_metrics(act_logits: torch.Tensor, target_is_active: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate F1, Precision, Recall, and Exact Match for phase activity."""
    # Handle both logits and binary activation
    if act_logits.dtype == torch.bool:
        act_pred = act_logits.float()
    elif (act_logits >= 0).all() and (act_logits <= 1).all() and act_logits.dtype != torch.bool:
        # Already probabilities
        act_pred = (act_logits > threshold).float()
    else:
        # Logits
        act_pred = (torch.sigmoid(act_logits) > threshold).float()
        
    target_is_active = target_is_active.float()
    tp = (act_pred * target_is_active).sum()
    fp = (act_pred * (1 - target_is_active)).sum()
    fn = ((1 - act_pred) * target_is_active).sum()
    
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    exact_match = (act_pred == target_is_active).all(dim=1).float().mean().item()
    
    return {'act_f1': f1, 'act_precision': precision, 'act_recall': recall, 'act_exact_match': exact_match}

def calculate_ranking_metrics(act_logits: torch.Tensor, best_perms: torch.Tensor, target_is_active: torch.Tensor) -> Dict[str, Any]:
    """Calculate Top-K Accuracy and Oracle Exact Match."""
    B, K = act_logits.shape
    _, sorted_indices = torch.sort(act_logits, dim=1, descending=True)
    # best_perms maps Pred Index -> Target Index
    matched_target_indices = torch.gather(best_perms, 1, sorted_indices)

    # Mask for valid indices (where a prediction was matched to a target)
    valid_match_mask = (matched_target_indices != -1)

    # Clamp indices to avoid out-of-bounds error. Invalid indices will gather from index 0,
    # but will be ignored later by the mask.
    clamped_indices = torch.clamp(matched_target_indices, min=0)

    # Gather using clamped indices
    is_hit_unsafe = torch.gather(target_is_active.bool(), 1, clamped_indices)

    # A "hit" is only valid if the prediction was matched to a target in the first place.
    is_hit = is_hit_unsafe & valid_match_mask
    
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
# 4. Retrieval-based Identification Metrics (Top-K)
# =============================================================================

def calculate_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Calculates cosine similarity between two batches of vectors."""
    # vec1: [N, L], vec2: [M, L] -> Result: [N, M]
    vec1_norm = F.normalize(vec1, p=2, dim=1)
    vec2_norm = F.normalize(vec2, p=2, dim=1)
    return torch.mm(vec1_norm, vec2_norm.t())

def calculate_retrieval_topk(
    separated_patterns: torch.Tensor, 
    true_phase_ids: torch.Tensor,
    active_mask: torch.Tensor,
    reference_library: torch.Tensor,
    reference_ids: List[int],
    top_ks: List[int] = [1, 3, 10]
) -> Dict[str, float]:
    """
    Correct Top-K Definition: 
    For each truly active phase, compare its separated slot pattern with 
    the external reference database using cosine similarity.
    """
    device = separated_patterns.device
    B, K, L = separated_patterns.shape
    
    # reference_library: [NumRefs, L]
    # reference_ids: List of IDs corresponding to library rows
    ref_id_to_idx = {rid: i for i, rid in enumerate(reference_ids)}
    
    hits = {f'top{k}_acc': 0.0 for k in top_ks}
    total_active_phases = 0
    
    # Flatten batches to process all active phases at once
    # active_mask: [B, K] Boolean
    active_patterns = separated_patterns[active_mask] # [NumActive, L]
    active_ids = true_phase_ids[active_mask].cpu().numpy() # [NumActive]
    
    if active_patterns.shape[0] == 0:
        return {f'top{k}_acc': 1.0 for k in top_ks}

    # Calculate similarity with entire reference library: [NumActive, NumRefs]
    similarities = calculate_cosine_similarity(active_patterns, reference_library)
    
    # Get top indices
    max_k = max(top_ks)
    _, top_indices = torch.topk(similarities, k=min(max_k, similarities.shape[1]), dim=1)
    top_indices = top_indices.cpu().numpy()
    
    for i in range(active_patterns.shape[0]):
        true_id = int(active_ids[i])
        if true_id not in ref_id_to_idx:
            continue # Should not happen if library is complete
            
        true_idx = ref_id_to_idx[true_id]
        total_active_phases += 1
        
        for k in top_ks:
            if true_idx in top_indices[i, :k]:
                hits[f'top{k}_acc'] += 1.0
                
    return {k: v / (total_active_phases + 1e-8) for k, v in hits.items()}

# =============================================================================
# 5. Quantitative Analysis Metrics
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
# 3. Alignment Utility
# =============================================================================

def align_predictions(pred_patterns: torch.Tensor, target_patterns: torch.Tensor):
    """
    Finds the optimal permutation to align predicted patterns with target patterns.
    pred_patterns: [B, K, L], target_patterns: [B, K, L]
    """
    B, K, L = pred_patterns.shape
    device = pred_patterns.device
    
    perms = list(itertools.permutations(range(K)))
    perms_tensor = torch.tensor(perms, device=device) # [N_PERMS, K]
    
    # Expand for batch calculation
    # target_patterns: [B, K, L] -> [B, 1, K, L]
    target_patterns_expanded = target_patterns.unsqueeze(1)
    
    # [B, N_PERMS, K, L]
    target_perms = torch.gather(
        target_patterns_expanded.expand(-1, len(perms), -1, -1), 
        2, 
        perms_tensor.view(1, len(perms), K, 1).expand(B, -1, -1, L)
    )
    
    # Calculate MSE for each permutation: [B, N_PERMS]
    mse = torch.mean((pred_patterns.unsqueeze(1) - target_perms)**2, dim=(2, 3))
    
    best_perm_idx = torch.argmin(mse, dim=1) # [B]
    best_perms = perms_tensor[best_perm_idx] # [B, K]
    
    return best_perms

# =============================================================================
# Global Wrapper for Separation Model
# =============================================================================

def calculate_all_metrics(
    pred_patterns: torch.Tensor, 
    target_patterns: torch.Tensor,
    phase_ids: torch.Tensor,
    reference_library: Optional[torch.Tensor] = None,
    reference_ids: Optional[List[int]] = None,
    active_threshold: float = 1e-4,
    two_theta_range: Tuple[float, float] = (10.0, 80.0),
    act_logits: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Calculate all metrics for the Separation Model, handling K_pred != K_target.
    """
    from scipy.optimize import linear_sum_assignment

    B, K_pred, L = pred_patterns.shape
    _, K_target, _ = target_patterns.shape
    device = pred_patterns.device

    p_active_list, t_active_list = [], []
    aligned_preds = torch.zeros_like(target_patterns)
    target_to_pred_map = torch.full((B, K_target), -1, dtype=torch.long, device=device)
    target_is_active = torch.sum(target_patterns**2, dim=-1) > active_threshold

    for i in range(B):
        active_target_mask = target_is_active[i]
        num_active_targets = active_target_mask.sum().item()
        if num_active_targets == 0:
            continue

        active_target_patterns = target_patterns[i, active_target_mask]
        cost_matrix = torch.cdist(pred_patterns[i], active_target_patterns, p=2) ** 2
        pred_indices, target_indices_relative = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        original_target_indices = torch.where(active_target_mask)[0].to(device)
        target_indices_absolute = original_target_indices[target_indices_relative]

        p_active_list.append(pred_patterns[i, pred_indices])
        t_active_list.append(target_patterns[i, target_indices_absolute])
        
        aligned_preds[i, target_indices_absolute] = pred_patterns[i, pred_indices]
        target_to_pred_map[i, target_indices_absolute] = torch.tensor(pred_indices, device=device, dtype=torch.long)

    results = {}

    pred_is_active = torch.zeros_like(target_is_active, dtype=torch.bool)
    if act_logits is not None:
        pred_is_active_unaligned = (torch.sigmoid(act_logits) > 0.5)
        for i in range(B):
            matched_pred_indices = target_to_pred_map[i][target_to_pred_map[i] != -1]
            if len(matched_pred_indices) > 0:
                 matched_target_indices = torch.where(target_to_pred_map[i] != -1)[0]
                 pred_is_active[i, matched_target_indices] = pred_is_active_unaligned[i, matched_pred_indices]
    else:
        pred_is_active = torch.sum(aligned_preds**2, dim=-1) > active_threshold
    results.update(calculate_activity_metrics(pred_is_active, target_is_active))

    pred_to_target_map = torch.full((B, K_pred), -1, dtype=torch.long, device=device)
    for i in range(B):
        for target_idx in range(K_target):
            pred_idx = target_to_pred_map[i, target_idx].item()
            if pred_idx != -1:
                pred_to_target_map[i, pred_idx] = target_idx
    
    pseudo_logits = act_logits if act_logits is not None else torch.sum(pred_patterns**2, dim=-1)
    rank_res = calculate_ranking_metrics(pseudo_logits, pred_to_target_map, target_is_active)
    results.update(rank_res)

    if not p_active_list:
        results.update({k: 0.0 for k in ['rwp', 'pearson', 'si_sdr', 'sir', 'sar', 'delta_2theta', 'fwhm_error', 'intensity_consistency']})
    else:
        p_active = torch.cat(p_active_list, dim=0)
        t_active = torch.cat(t_active_list, dim=0)
        sir, sar = calculate_sir_sar(p_active, t_active)
        results.update({
            'rwp': calculate_rwp(p_active, t_active),
            'pearson': calculate_pearson_correlation(p_active, t_active),
            'si_sdr': calculate_sisdr(p_active, t_active),
            'sir': sir, 'sar': sar,
            'delta_2theta': calculate_peak_shift_delta_2theta(p_active, t_active, two_theta_range),
            'fwhm_error': calculate_fwhm_error(p_active, t_active),
            'intensity_consistency': calculate_intensity_ratio_consistency(p_active, t_active)
        })

    if reference_library is not None and reference_ids is not None:
        ret_metrics = calculate_retrieval_topk(
            aligned_preds, phase_ids, target_is_active, 
            reference_library, reference_ids
        )
        results.update(ret_metrics)
        
    quant_res = calculate_quantitative_metrics(aligned_preds, target_patterns)
    results['quant_mae'] = quant_res['quant_mae']
    
    return results

# =============================================================================
# Loss Functions
# =============================================================================

class SeparationLoss(nn.Module):
    """
    PIT (Permutation Invariant Training) Loss for Multi-source Separation.
    Calculates the minimum MSE across all possible permutations of predictions.
    """
    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: [B, K, L]
        target: [B, K, L]
        """
        B, K, L = target.shape
        device = target.device
        
        # 1. 生成所有可能的排列
        perms = list(itertools.permutations(range(K)))
        perms_tensor = torch.tensor(perms, device=device) # [N_PERMS, K]
        num_perms = len(perms)
        
        # 2. 对目标进行所有排列展开: [B, N_PERMS, K, L]
        target_expanded = target.unsqueeze(1).expand(-1, num_perms, -1, -1)
        # 利用 gather 根据排列索引获取对应的目标分量
        # perms_tensor: [N_PERMS, K] -> [B, N_PERMS, K, 1] -> [B, N_PERMS, K, L]
        idx = perms_tensor.view(1, num_perms, K, 1).expand(B, -1, -1, L)
        target_perms = torch.gather(target_expanded, 2, idx)
        
        # 3. 计算预测值与所有排列目标之间的 MSE: [B, N_PERMS]
        # pred: [B, K, L] -> [B, 1, K, L]
        mse_all = torch.mean((pred.unsqueeze(1) - target_perms)**2, dim=(2, 3))
        
        # 4. 对每个样本取最小 MSE，然后在 Batch 上求平均
        min_mse, _ = torch.min(mse_all, dim=1)
        return torch.mean(min_mse)
