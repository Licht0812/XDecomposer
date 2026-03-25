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
    if pred_pattern.dim() == 1:
        pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    target_pattern, pred_pattern = torch.clamp(target_pattern, min=0), torch.clamp(pred_pattern, min=0)
    diff_sq = (target_pattern - pred_pattern) ** 2
    numerator = torch.sum(diff_sq, dim=-1)
    denominator = torch.sum(target_pattern ** 2, dim=-1) + epsilon
    return torch.sqrt(numerator / denominator).mean().item()

def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    if pred_pattern.dim() == 1:
        pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    pred_mean, target_mean = pred_pattern.mean(dim=-1, keepdim=True), target_pattern.mean(dim=-1, keepdim=True)
    pred_centered, target_centered = pred_pattern - pred_mean, target_pattern - target_mean
    numerator = (pred_centered * target_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)
    return (numerator / (pred_std * target_std + 1e-8)).mean().item()

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
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

# =============================================================================
# 2. Activity / Presence Metrics (Multi-label Binary Classification)
# =============================================================================

def calculate_activity_metrics(pred_active: torch.Tensor, target_active: torch.Tensor) -> Dict[str, float]:
    """
    Calculates Precision, Recall, F1, and Exact Match for phase presence.
    pred_active, target_active: [B, K] Boolean Tensors
    """
    tp = torch.logical_and(pred_active, target_active).sum().float()
    fp = torch.logical_and(pred_active, torch.logical_not(target_active)).sum().float()
    fn = torch.logical_and(torch.logical_not(pred_active), target_active).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Exact Match: Every slot in the sample must match exactly
    exact_match = (pred_active == target_active).all(dim=1).float().mean()
    
    return {
        'act_precision': precision.item(),
        'act_recall': recall.item(),
        'act_f1': f1.item(),
        'act_exact_match': exact_match.item()
    }

# =============================================================================
# 3. Retrieval-based Identification Metrics (Top-K)
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
    active_threshold: float = 1e-4
) -> Dict[str, float]:
    """
    Calculate all metrics for the Separation Model.
    """
    B, K, L = target_patterns.shape
    
    # 1. Alignment
    best_perms = align_predictions(pred_patterns, target_patterns)
    
    # Re-order pred_patterns to match target_patterns slots
    best_perms_expanded = best_perms.unsqueeze(-1).expand(-1, -1, L)
    aligned_preds = torch.gather(pred_patterns, 1, best_perms_expanded)
    
    # 2. Activity / Presence Metrics (based on energy)
    # Target is active if energy > threshold
    target_active = torch.sum(target_patterns**2, dim=-1) > active_threshold
    # Prediction is active if energy > threshold
    pred_active = torch.sum(aligned_preds**2, dim=-1) > active_threshold
    
    results = calculate_activity_metrics(pred_active, target_active)
    
    # 3. Separation Waveform Metrics (only for truly active slots)
    if not target_active.any():
        # Fallback if no active phases in batch (should not happen with MIN_K=2)
        results.update({k: 0.0 for k in ['rwp', 'pearson', 'si_sdr']})
    else:
        p_active = aligned_preds[target_active]
        t_active = target_patterns[target_active]
        
        results.update({
            'rwp': calculate_rwp(p_active, t_active),
            'pearson': calculate_pearson_correlation(p_active, t_active),
            'si_sdr': calculate_sisdr(p_active, t_active),
        })
    
    # 4. Retrieval Top-K Metrics
    if reference_library is not None and reference_ids is not None:
        ret_metrics = calculate_retrieval_topk(
            aligned_preds, phase_ids, target_active, 
            reference_library, reference_ids
        )
        results.update(ret_metrics)
        
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
