import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from model.dataset import ASEDataset, OnlineMixingConfig, RRUFFOnlineMixingDataset
from model.XQueryer import Xmodel

from typing import Dict, Tuple, Optional, Any, List

# =============================================================================
# Standard Metrics from reference/metrics.py (Integrated)
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
    # 伪影 = 预测 - 真实目标
    artifacts = pred - target
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    artifacts_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    # SAR：信号 / 伪影
    sar = 10 * torch.log10(target_energy / artifacts_energy)
    
    # SIR：信号 / 干扰（多相场景下，interference是其他相位的XRD之和，需外部传入）
    if interference is None:
        raise ValueError("多相场景下必须传入interference（其他相位的XRD pattern之和）")
    if interference.ndim == 1:
        interference = interference.unsqueeze(0)
    interference_energy = torch.sum(interference ** 2, dim=-1) + eps
    sir = 10 * torch.log10(target_energy / interference_energy)
    
    return sir.mean().item(), sar.mean().item()

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

def calculate_peak_shift_delta_2theta(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, two_theta_range: Tuple[float, float] = (10.0, 80.0), tolerance: int = 10, **peak_args: Any) -> float:
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

def calculate_activity_metrics(act_logits: torch.Tensor, target_is_active: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate F1, Precision, Recall, and Exact Match for phase activity."""
    if act_logits.dtype == torch.bool:
        act_pred = act_logits.float()
    else:
        act_pred = (torch.sigmoid(act_logits) > threshold).float()
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
    matched_target_indices = torch.gather(best_perms, 1, sorted_indices)
    is_hit = torch.gather(target_is_active, 1, matched_target_indices) # [B, K]
    num_actives = target_is_active.sum(dim=1).long()
    oracle_hits = 0
    topk_acc = {f'top{k}_acc': 0.0 for k in [1, 10]}
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

def calculate_quantitative_metrics(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> Dict[str, Any]:
    """Calculate Quantitative Mean Absolute Error (Quant MAE) in percentage points."""
    if pred_patterns.dim() == 2:
        pred_patterns = pred_patterns.unsqueeze(0)
        target_patterns = target_patterns.unsqueeze(0)
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return {'quant_mae': mae, 'pred_pct': pred_pct, 'target_pct': target_pct}

def run_one_epoch(model, dataloader, device, entries_dict, threshold=0.3, save_path='inference_results.json', limit=0):
    model.eval()
    total_to_eval = len(dataloader.dataset) if limit == 0 else min(limit, len(dataloader.dataset))
    pbar = tqdm(total=total_to_eval, desc='Evaluating... ', unit='data')
    
    metrics = {
        'total_phases': 0,
        'top1_hits': 0,
        'top5_hits': 0,
        'top10_hits': 0,
        'rwp_sum': 0,
        'pearson_sum': 0,
        'sisdr_sum': 0,
        'sir_sum': 0,
        'sar_sum': 0,
        'delta_2theta_sum': 0,
        'fwhm_error_sum': 0,
        'intensity_consistency_sum': 0,
        'ratio_mae_sum': 0,
        'act_f1_sum': 0,
        'act_precision_sum': 0,
        'act_recall_sum': 0,
        'act_exact_match_sum': 0,
        'oracle_exact_match_sum': 0,
        'total_samples': 0
    }

    # Use cpu autocast if cuda not available
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Open file for streaming results
    with open(save_path, 'w') as f:
        f.write('[\n') 

        for batch_idx, batch in enumerate(dataloader):
            if limit > 0 and metrics['total_samples'] >= limit:
                break
            
            intensity = batch['intensity'].to(device)
            element = batch['element'].to(device)
            gt_xrds = batch['gt_xrds'].to(device)
            gt_ratios = batch['gt_ratios'].to(device)
            gt_ids = batch['gt_ids'].to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type):
                    outputs = model(intensity, element)
                    pred_xrds = outputs['xrds']
                    pred_ratios = outputs['ratios']
                    feat_logits = outputs['feat_logits']

            # Process batch
            for b in range(intensity.size(0)):
                valid_gt_mask = gt_ratios[b] > 1e-6
                num_valid_gt = valid_gt_mask.sum().item()
                if num_valid_gt == 0: continue
                
                # Hungarian Matching
                # Convert to float32 as cdist might not support float16 on some platforms/versions
                cost_matrix = torch.cdist(pred_xrds[b].float(), gt_xrds[b][valid_gt_mask].float(), p=2).cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                sample_results = {
                    "sample_id": metrics['total_samples'],
                    "phases": []
                }
                
                for r, c in zip(row_ind, col_ind):
                    gt_idx = torch.where(valid_gt_mask)[0][c]
                    target_id = int(gt_ids[b, gt_idx].item())
                    
                    # Top-K Retrieval Metrics
                    logits = feat_logits[b, r]
                    top10_indices = torch.topk(logits, k=10).indices.cpu().numpy()
                    
                    metrics['total_phases'] += 1
                    if target_id == top10_indices[0]: metrics['top1_hits'] += 1
                    if target_id in top10_indices[:5]: metrics['top5_hits'] += 1
                    if target_id in top10_indices[:10]: metrics['top10_hits'] += 1
                    
                    # Reconstruction Metrics
                    rwp = calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    pearson = calculate_pearson_correlation(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    sisdr = calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    
                    other_phases_mask = valid_gt_mask.clone()
                    other_phases_mask[gt_idx] = False
                    interference = gt_xrds[b][other_phases_mask].sum(dim=0) if other_phases_mask.any() else torch.zeros_like(gt_xrds[b, gt_idx])
                    
                    sir, sar = calculate_sir_sar(pred_xrds[b, r], gt_xrds[b, gt_idx], interference=interference)
                    delta_2theta = calculate_peak_shift_delta_2theta(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    fwhm_error = calculate_fwhm_error(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    intensity_consistency = calculate_intensity_ratio_consistency(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    
                    metrics['rwp_sum'] += rwp
                    metrics['pearson_sum'] += pearson
                    metrics['sisdr_sum'] += sisdr
                    metrics['sir_sum'] += sir
                    metrics['sar_sum'] += sar
                    metrics['delta_2theta_sum'] += delta_2theta
                    metrics['fwhm_error_sum'] += fwhm_error
                    metrics['intensity_consistency_sum'] += intensity_consistency
                    
                    phase_data = {
                        "gt_id": target_id,
                        "gt_mpid": entries_dict.get(str(target_id), {}).get("mpid", "Unknown"),
                        "gt_ratio": float(gt_ratios[b, gt_idx].item()),
                        "pred_ratio": float(pred_ratios[b, r].item()),
                        "rwp": rwp,
                        "pearson": pearson,
                        "sisdr": sisdr,
                        "sir": sir,
                        "sar": sar,
                        "delta_2theta": delta_2theta,
                        "fwhm_error": fwhm_error,
                        "intensity_consistency": intensity_consistency,
                        "top10_preds": [
                            {
                                "id": int(idx),
                                "mpid": entries_dict.get(str(int(idx)), {}).get("mpid", "Unknown")
                            } for idx in top10_indices
                        ]
                    }
                    sample_results["phases"].append(phase_data)
                
                # Compute Quant MAE for the sample
                matched_pred_xrds = []
                matched_gt_xrds = []
                for r, c in zip(row_ind, col_ind):
                    gt_idx = torch.where(valid_gt_mask)[0][c]
                    matched_pred_xrds.append(pred_xrds[b, r])
                    matched_gt_xrds.append(gt_xrds[b, gt_idx])
                
                if matched_pred_xrds:
                    quant_res = calculate_quantitative_metrics(torch.stack(matched_pred_xrds), torch.stack(matched_gt_xrds))
                    q_mae = quant_res['quant_mae']
                    metrics['ratio_mae_sum'] += q_mae * len(matched_pred_xrds) # accumulate by number of phases to average correctly later
                    
                    # Update the JSON output to include the calculated quant_pct
                    for i, phase_data in enumerate(sample_results["phases"]):
                        phase_data["pred_pct"] = float(quant_res['pred_pct'][0, i].item())
                        phase_data["target_pct"] = float(quant_res['target_pct'][0, i].item())

            # Sample-level Activity Metrics
            # Target is active if gt_ratio > 1e-6
            # Pred is active if pred_ratio > 1e-6 (or use a threshold)
            target_is_active = (gt_ratios[b] > 1e-6).float()
            # To calculate sample-level act metrics, we need to know which slot corresponds to which gt
            # We already have row_ind and col_ind from Hungarian matching
            # Let's use the matched row_ind to determine pred_is_active
            pred_is_active = torch.zeros_like(target_is_active)
            for r, c in zip(row_ind, col_ind):
                gt_idx = torch.where(valid_gt_mask)[0][c]
                if pred_ratios[b, r] > threshold:
                    pred_is_active[gt_idx] = 1.0
            
            act_res = calculate_activity_metrics(pred_is_active.unsqueeze(0), target_is_active.unsqueeze(0))
            metrics['act_f1_sum'] += act_res['act_f1']
            metrics['act_precision_sum'] += act_res['act_precision']
            metrics['act_recall_sum'] += act_res['act_recall']
            metrics['act_exact_match_sum'] += act_res['act_exact_match']
            
            # Oracle Exact Match
            # Use feat_logits as pseudo-logits for ranking
            # We need to map slots back to GT order for calculate_ranking_metrics
            # But XQueryer's slots are already matched to GTs via Hungarian matching
            # Let's simplify: if all GTs are correctly identified in their matched slots
            best_perms = torch.tensor(row_ind).unsqueeze(0) # Not exactly what's needed for calculate_ranking_metrics
            # Let's just use the logic directly
            is_all_matched = True
            for r, c in zip(row_ind, col_ind):
                gt_idx = torch.where(valid_gt_mask)[0][c]
                target_id = int(gt_ids[b, gt_idx].item())
                top1_id = torch.argmax(feat_logits[b, r]).item()
                if target_id != top1_id:
                    is_all_matched = False
                    break
            if is_all_matched: metrics['oracle_exact_match_sum'] += 1.0

            metrics['total_samples'] += 1
            if metrics['total_samples'] > 1: f.write(',\n')
            json.dump(sample_results, f)

            pbar.update(intensity.size(0))
            
            # Memory Management
            if metrics['total_samples'] % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            if metrics['total_samples'] % 100 == 0:
                avg_t10 = metrics['top10_hits'] / metrics['total_phases'] if metrics['total_phases'] > 0 else 0
                pbar.set_postfix(top10=f"{avg_t10:.4f}")

        f.write('\n]') 

    pbar.close()
    
    print("\n--- Final Evaluation Summary ---")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Total Phases:  {metrics['total_phases']}")
    if metrics['total_phases'] > 0:
        print(f"Top-1 Accuracy:  {metrics['top1_hits']/metrics['total_phases']:.4f}")
        print(f"Top-5 Accuracy:  {metrics['top5_hits']/metrics['total_phases']:.4f}")
        print(f"Top-10 Accuracy: {metrics['top10_hits']/metrics['total_phases']:.4f}")
        print(f"Avg RWP:         {metrics['rwp_sum']/metrics['total_phases']:.4f}")
        print(f"Avg Pearson:     {metrics['pearson_sum']/metrics['total_phases']:.4f}")
        print(f"Avg SI-SDR:      {metrics['sisdr_sum']/metrics['total_phases']:.2f} dB")
        print(f"Avg SIR:         {metrics['sir_sum']/metrics['total_phases']:.2f} dB")
        print(f"Avg SAR:         {metrics['sar_sum']/metrics['total_phases']:.2f} dB")
        print(f"Avg Δ2θ:         {metrics['delta_2theta_sum']/metrics['total_phases']:.4f}")
        print(f"Avg FWHM Err:    {metrics['fwhm_error_sum']/metrics['total_phases']:.4f}")
        print(f"Avg Int Consist: {metrics['intensity_consistency_sum']/metrics['total_phases']:.4f}")
        print(f"Avg Ratio MAE:   {metrics['ratio_mae_sum']/metrics['total_phases']*100:.2f}%")
        
    if metrics['total_samples'] > 0:
        print(f"Act F1:          {metrics['act_f1_sum']/metrics['total_samples']:.4f}")
        print(f"Act Precision:   {metrics['act_precision_sum']/metrics['total_samples']:.4f}")
        print(f"Act Recall:      {metrics['act_recall_sum']/metrics['total_samples']:.4f}")
        print(f"Act Exact Match: {metrics['act_exact_match_sum']/metrics['total_samples']:.4f}")
        print(f"Oracle Match:    {metrics['oracle_exact_match_sum']/metrics['total_samples']:.4f}")
    print(f"Results saved to: {save_path}")

    return None

def main():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Ensure entries_dict path is correctly resolved relative to the script if default is used
    entries_dict_path = args.entries_dict
    if entries_dict_path == './entries_dict.json':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        entries_dict_path = os.path.join(script_dir, 'entries_dict.json')
    
    # Load entries dictionary for crystal info retrieval
    with open(entries_dict_path, 'r') as f:
        entries_dict = json.load(f)
    print(f"Loaded crystal entries from {entries_dict_path}")

    model = Xmodel(embed_dim=3500, num_slots=args.num_slots, feature_dim=args.feature_dim, num_classes=args.num_classes)
    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print('Loaded model from {}'.format(args.load_path))

    # Configure data loader based on num_phases
    if args.num_phases in [2, 3, 4]:
        print(f"Configuring dataset for fixed {args.num_phases} phases.")
        min_k = args.num_phases
        max_k = args.num_phases
    else:
        print("Configuring dataset for random 2-4 phases.")
        min_k = 2
        max_k = 4

    testset = RRUFFOnlineMixingDataset(args.db_path, split='test', num_folds=args.num_folds, fold=args.fold,
                         encode_element=args.atom_embed, num_classes=args.num_classes,
                         min_k=min_k, max_k=max_k)
    
    # Use smaller batch size if on CPU to prevent OOM
    batch_size = args.batch_size if torch.cuda.is_available() else 1
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # Define output file path
    output_dir = os.path.dirname(args.load_path)
    if 'checkpoints' in output_dir:
        output_dir = os.path.dirname(output_dir)
    if args.num_phases > 0:
        save_path = os.path.join(output_dir, f'inference_results_{args.num_phases}_phases.json')
    else:
        save_path = os.path.join(output_dir, 'inference_results.json')
    print(f"Saving results to {save_path}")

    run_one_epoch(model, test_loader, device, entries_dict, threshold=args.threshold, limit=args.limit, save_path=save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, choices=['cuda:0', 'cpu'])
    parser.add_argument('--db_path', default='/data/group/project1/Crystal/UniqCryLabeled.db', type=str)
    parser.add_argument('--npz_dir', default='/data/group/project1/Crystal/UniqCry', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--atom_embed', default=True, type=bool)
    parser.add_argument('--load_path', default='/home/cb/XRDS/XQueryer/output/2024-09-09_1117/checkpoints/checkpoint_0010.pth', type=str)
    parser.add_argument('--num_classes', default=100315, type=int)
    parser.add_argument('--num_slots', default=4, type=int)
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--entries_dict', default='./entries_dict.json', type=str)
    parser.add_argument('--threshold', default=0.3, type=float, help='Threshold for phase identification')
    parser.add_argument('--limit', default=0, type=int, help='Limit number of samples to evaluate (0 for all)')
    parser.add_argument('--num_phases', type=int, default=0, help='Specify a fixed number of phases (2, 3, or 4). Default 0 for random mix.')
    parser.add_argument('--fold', default=0, type=int, help='Current fold index')
    parser.add_argument('--num_folds', default=5, type=int, help='Total number of folds')

    args = parser.parse_args()
    main()
    print('THE END')
