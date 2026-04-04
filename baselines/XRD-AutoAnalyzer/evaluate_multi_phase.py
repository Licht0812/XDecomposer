import os
import sqlite3
import pickle
import random
import numpy as np
import tensorflow as tf
from autoXRD.spectrum_analysis import SpectrumAnalyzer
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DB_PATH = "/data/group/project1/Crystal/UniqCryLabeled.db"
NPZ_DIR = "/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data"
REF_DIR = "Novel-Space/References"
MODEL_PATH = "Model_Reconstructor.h5"
MAPPING_PATH = "id_to_ref_mapping_full.pkl"
FINGERPRINT_PATH = "reference_fingerprints_full.npy"
OUTPUT_DIR = "evaluation_results_multiphase"
TARGET_LENGTH = 3500
SEED = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    artifacts = pred - target
    target_energy, artifacts_energy = torch.sum(target ** 2, dim=-1) + eps, torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)
    interference = interference if interference is not None else artifacts
    if interference.ndim == 1: interference = interference.unsqueeze(0)
    sir = 10 * torch.log10(target_energy / (torch.sum(interference ** 2, dim=-1) + eps))
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
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return float('nan')
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
    if not matched_mask.any(): return float('nan')
    return (min_dists[matched_mask].sum() / matched_mask.sum()).item()

def calculate_fwhm_error(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, window_size: int = 50, **peak_args: Any) -> float:
    """Calculate FWHM error (Vectorized)."""
    if pred_pattern.dim() == 1: pred_pattern, target_pattern = pred_pattern.unsqueeze(0), target_pattern.unsqueeze(0)
    B, L = pred_pattern.shape
    device = pred_pattern.device
    p_mask, t_mask = find_xrd_peaks_batch(pred_pattern, **peak_args), find_xrd_peaks_batch(target_pattern, **peak_args)
    p_idxs, t_idxs = _get_batch_peak_indices(p_mask), _get_batch_peak_indices(t_mask)
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return float('nan')
    dists = torch.abs(p_idxs.unsqueeze(2).float() - t_idxs.unsqueeze(1).float())
    final_mask = (p_idxs != -1).unsqueeze(2) & (t_idxs != -1).unsqueeze(1) & (dists <= 10)
    min_dists, min_t_indices_rel = dists.masked_fill(~final_mask, float('inf')).min(dim=2)
    matched_mask = min_dists != float('inf')
    if not matched_mask.any(): return float('nan')
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
    if p_idxs.shape[1] == 0 or t_idxs.shape[1] == 0: return float('nan')
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
    pred_intensities = torch.sum(torch.clamp(pred_patterns, min=0), dim=-1)
    target_intensities = torch.sum(target_patterns, dim=-1)
    pred_pct = pred_intensities / (pred_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_intensities / (target_intensities.sum(dim=-1, keepdim=True) + 1e-8)
    mae = torch.abs(pred_pct - target_pct).mean().item() * 100
    return {'quant_mae': mae}

def calculate_all_metrics_wrapper(pred_patterns, target_patterns, gt_refs, pred_refs, act_logits=None, best_perms=None):
    """Wrapper to calculate all reference metrics for a single sample."""
    # Convert to torch tensors
    p = torch.from_numpy(pred_patterns).float()
    t = torch.from_numpy(target_patterns).float()
    if p.dim() == 2: p, t = p.unsqueeze(0), t.unsqueeze(0)
    
    # 1. Separation Metrics (only for active phases)
    is_active = torch.sum(t ** 2, dim=-1) > 1e-6
    p_active, t_active = p[is_active], t[is_active]
    
    results = {}
    if is_active.any():
        sir, sar = calculate_sir_sar(p_active, t_active)
        results.update({
            'rwp': calculate_rwp(p_active, t_active),
            'pearson': calculate_pearson_correlation(p_active, t_active),
            'si_sdr': calculate_sisdr(p_active, t_active),
            'sir': sir, 'sar': sar,
            'delta_2theta': calculate_peak_shift_delta_2theta(p_active, t_active),
            'fwhm_error': calculate_fwhm_error(p_active, t_active),
            'intensity_consistency': calculate_intensity_ratio_consistency(p_active, t_active)
        })
    else:
        results.update({k: 0.0 for k in ['rwp', 'pearson', 'si_sdr', 'sir', 'sar', 'delta_2theta', 'fwhm_error', 'intensity_consistency']})
        
    # 2. Activity Metrics
    if act_logits is not None:
        target_is_active = is_active.float()
        results.update(calculate_activity_metrics(act_logits, target_is_active))
        if best_perms is not None:
            results.update(calculate_ranking_metrics(act_logits, best_perms, target_is_active))
            
    # 3. Quantitative Metrics
    results.update(calculate_quantitative_metrics(p, t))
    return results

def calculate_pearson(pred, target):
    return calculate_pearson_correlation(torch.from_numpy(pred).float(), torch.from_numpy(target).float())

def load_mapping():
    with open(MAPPING_PATH, 'rb') as f: return pickle.load(f)

def get_test_cids_aligned(mapping):
    all_ids = sorted(list(mapping.keys()))
    random.seed(SEED); np.random.seed(SEED)
    shuffled = all_ids.copy(); random.shuffle(shuffled)
    return shuffled[int(len(shuffled) * 0.9):]

def load_npz(label, s_idx):
    row_id = label + 1
    fpath = os.path.join(NPZ_DIR, f"crystal_{row_id}_sample_{s_idx:02d}.npz")
    try:
        data = np.load(fpath)
        y = data['y'] if 'y' in data else data['intensity']
        if len(y) != TARGET_LENGTH:
            y = np.interp(np.linspace(10, 80, TARGET_LENGTH), np.linspace(10, 80, len(y)), y)
        return y.astype(np.float32)
    except: return None

class EvaluatorReferenceBank:
    def __init__(self):
        self.fingerprints = torch.from_numpy(np.load(FINGERPRINT_PATH)).float()
        with open('class_list.pkl', 'rb') as f:
            self.ref_names = [n[:-4] if n.endswith('.cif') else n for n in pickle.load(f)]
    def get_topk(self, query_waveform, k=10):
        t_query = torch.from_numpy(query_waveform).float().view(1, -1)
        sims = F.cosine_similarity(t_query, self.fingerprints)
        topk_vals, topk_idxs = torch.topk(sims, k)
        return [self.ref_names[i] for i in topk_idxs.tolist()]

def generate_mixed_sample_with_anchor(anchor_cid, all_test_cids, mapping, k):
    # k = random.randint(2, 4)
    if k > 1:
        others = random.sample([c for c in all_test_cids if c != anchor_cid], k - 1)
    else:
        others = []
    selected_cids = [anchor_cid] + others
    random.shuffle(selected_cids)
    mixed_y_unscaled = np.zeros(TARGET_LENGTH, dtype=np.float32)
    gt_components = []
    weights = np.random.dirichlet(np.ones(k))
    for i, cid in enumerate(selected_cids):
        y = load_npz(cid, random.randint(0, 19))
        if y is None: continue
        y /= (np.max(y) + 1e-8)
        mixed_y_unscaled += weights[i] * y
        gt_components.append({'ref': mapping[cid][:-4], 'weight': weights[i], 'y': y})
    
    scale_factor = 1.0
    if np.max(mixed_y_unscaled) > 0: 
        scale_factor = 100.0 / np.max(mixed_y_unscaled)
        mixed_y = mixed_y_unscaled * scale_factor
    else:
        mixed_y = mixed_y_unscaled
        
    for comp in gt_components:
        comp['y'] = comp['y'] * comp['weight'] * scale_factor
        
    return mixed_y, gt_components

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-phase XRD analysis for a specific number of phases.")
    parser.add_argument("--num_phases", type=int, required=True, choices=[2, 3, 4], help="The number of phases in the mixtures to evaluate.")
    args = parser.parse_args()

    mapping = load_mapping(); test_cids = get_test_cids_aligned(mapping); ref_bank = EvaluatorReferenceBank()
    print(f"Total test crystals: {len(test_cids)}")
    all_results = []
    for anchor_id in tqdm(test_cids, desc=f"Evaluating for {args.num_phases} phases"):
        mixed_y, gt_components = generate_mixed_sample_with_anchor(anchor_id, test_cids, mapping, args.num_phases)
        if not gt_components: continue
        xy_path = f"temp_eval_{anchor_id}.xy"
        np.savetxt(xy_path, np.column_stack((np.linspace(10, 80, TARGET_LENGTH), mixed_y)))
        try:
            analyzer = SpectrumAnalyzer(spectra_dir=".", spectrum_fname=xy_path, max_phases=4, cutoff_intensity=0.5, min_conf=0.85, reference_dir=REF_DIR, model_path=MODEL_PATH)
            mixtures, confidences, _, scalings, spectra = analyzer.suspected_mixtures
            os.remove(xy_path)
            pred_refs, pred_weights, pred_waveforms = [], [], []
            if mixtures and mixtures[0]:
                pred_refs = [r[:-4] for r in mixtures[0]]
                pred_weights = [w/sum(scalings[0]) for w in scalings[0]]
                pred_waveforms = spectra[0]
            # Alignment (for metrics calculation)
            all_gt_refs = [c['ref'] for c in gt_components]
            gt_patterns = np.zeros((4, TARGET_LENGTH))
            for i, comp in enumerate(gt_components): gt_patterns[i] = comp['y']
            
            # Align predictions with targets based on names if possible
            aligned_pred_patterns = np.zeros((4, TARGET_LENGTH))
            pred_weights_padded = [0.0] * 4
            for i, gt_ref in enumerate(all_gt_refs):
                if gt_ref in pred_refs:
                    p_idx = pred_refs.index(gt_ref)
                    aligned_pred_patterns[i] = pred_waveforms[p_idx] * scalings[0][p_idx]
                    pred_weights_padded[i] = pred_weights[p_idx]
            
            # Use pseudo-logits for activity metrics
            act_logits = torch.zeros((1, 4))
            for i, ref in enumerate(pred_refs):
                if i < 4: act_logits[0, i] = 10.0 # High value for predicted phases
            
            best_perms = torch.arange(4).unsqueeze(0) # Already aligned by name
            
            sample_metrics = calculate_all_metrics_wrapper(
                aligned_pred_patterns, gt_patterns, all_gt_refs, pred_refs, 
                act_logits=act_logits, best_perms=best_perms
            )
            
            # Retrieval metrics (ID Accuracy)
            top1_hits, top5_hits, top10_hits = 0, 0, 0
            for gt_ref in all_gt_refs:
                h1, h5, h10 = False, False, False
                for p_wave in pred_waveforms:
                    tk = ref_bank.get_topk(p_wave, k=10)
                    if gt_ref == tk[0]: h1 = True
                    if gt_ref in tk[:5]: h5 = True
                    if gt_ref in tk[:10]: h10 = True
                if h1: top1_hits += 1
                if h5: top5_hits += 1
                if h10: top10_hits += 1
            
            sample_metrics.update({
                'anchor_id': anchor_id,
                'num_phases': args.num_phases,
                'id_acc_top1': top1_hits/len(all_gt_refs),
                'id_acc_top5': top5_hits/len(all_gt_refs),
                'id_acc_top10': top10_hits/len(all_gt_refs)
            })
            all_results.append(sample_metrics)
        except Exception as e:
            if os.path.exists(xy_path): os.remove(xy_path)
            print(f"Error anchor {anchor_id}: {e}")
    if all_results:
        df = pd.DataFrame(all_results)
        summary = df.drop(columns=['anchor_id']).mean()
        print(f"\n--- Evaluation Summary for {args.num_phases} phases ---")
        print(summary)
        df.to_csv(os.path.join(OUTPUT_DIR, f"full_test_results_{args.num_phases}_phases.csv"), index=False)
if __name__ == "__main__": main()
