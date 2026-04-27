import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any

# Adjust sys.path to include root project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Models
from src.models.xrd_transformer import XRDMaskedAutoencoder
from src.models.xdecomposer import build_xdecomposer

# Data
from src.data.online_mixing_dataset import create_online_mixing_dataloader
from src.data.config import OnlineMixingConfig
from src.data.core import XRDBaseDataset, process_pattern

# Utils
from src.utils.metrics import calculate_separation_metrics
from src.utils.run_outputs import current_timestamp, extract_timestamp
from src.losses import calculate_pit_loss

# External libs for Crystal Structure & XRD
try:
    from ase.db import connect
    from ase.spacegroup import get_spacegroup
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    HAS_CRYSTAL_LIBS = True
except ImportError:
    HAS_CRYSTAL_LIBS = False
    print("Warning: ase or pymatgen not found. Theoretical XRD patterns will not be plotted.")

def build_reference_bank(dataset, device):
    """
    Builds a reference bank of XRD patterns from the dataset's available indices.
    Used for retrieval-based identification metrics.
    """
    # Simply use a cached attribute if available to avoid rebuilding
    if hasattr(dataset, '_cached_ref_bank'):
        return dataset._cached_ref_bank
        
    import os
    import torch
    
    # Safely get a path for saving the cache
    save_dir = getattr(dataset, 'data_dir', './data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = "rruff" if hasattr(dataset, 'phases') else "mp20"
    split_name = getattr(dataset, 'split', 'test')
    cache_path = os.path.join(save_dir, f"{dataset_name}_{split_name}_ref_bank_cache.pt")
    
    if os.path.exists(cache_path):
        print(f"Fast loading cached Reference Bank from {cache_path}...")
        ref_data = torch.load(cache_path, map_location=device)
        dataset._cached_ref_bank = ref_data
        return ref_data[0], ref_data[1]

    print(f"Building Reference Bank for Identification ({dataset_name})...")
    ref_patterns = []
    ref_ids = []
    count = 0
    
    if dataset_name == "rruff":
        # RRUFF dataset parsing from .phases directly
        for s_id, rruff_id, intensities in tqdm(dataset.phases, desc="Loading Refs (RRUFF)"):
            tensor = torch.tensor(intensities, dtype=torch.float32)
            if getattr(tensor, 'max')() > 0:
                tensor = tensor / tensor.max()
            ref_patterns.append(tensor)
            ref_ids.append(int(s_id))
            count += 1
            
    else:
        # MP20 dataset parsing
        # The OnlineMixingXRDDataset has a .indices attribute containing valid IDs for this split
        valid_ids = dataset.indices
        
        # Use the underlying get_crystal_patterns method
        # We need to access the base XRDBaseDataset method, which is available on dataset instance
        
        count = 0
        for cid in tqdm(valid_ids, desc="Loading Refs"):
            # Load 1 sample per ID (canonical pattern)
            patterns = dataset.get_crystal_patterns(int(cid), max_samples=1)
            if not patterns:
                continue
                
            # Process: Pad/Crop/Normalize
            # Note: We must match the normalization used in training/testing (usually Max scaling)
            # OnlineMixingDataset does Max scaling per sample.
            # Here we process the single reference similarly.
            
            # Use process_pattern from core.py
            # norm_method="max" ensures peak is at 1.0, consistent with model output range roughly
            tensor = process_pattern(patterns[0], dataset.xrd_length, norm_method="max")
            
            ref_patterns.append(tensor)
            ref_ids.append(int(cid))
            count += 1
        
    if count == 0:
        print("Warning: No reference patterns loaded!")
        return None, None
        
    ref_patterns = torch.stack(ref_patterns).to(device) # [M, L]
    ref_ids = torch.tensor(ref_ids, device=device)      # [M]
    
    print(f"Reference Bank built: {len(ref_ids)} patterns. Saving to cache...")
    torch.save((ref_patterns, ref_ids), cache_path)
    print(f"Cache saved locally to {cache_path}")
    dataset._cached_ref_bank = (ref_patterns, ref_ids)
    return ref_patterns, ref_ids

def calculate_identification_metrics(pred_patterns, target_ids, ref_bank, ref_ids, topk_list=[1, 3, 5]):
    """
    Calculates Top-K Identification Accuracy by retrieving most similar patterns from Ref Bank.
    
    Args:
        pred_patterns: [B, K, L] - Separated patterns
        target_ids: [B, K] - Ground Truth IDs
        ref_bank: [M, L] - Reference patterns
        ref_ids: [M] - Reference IDs
        topk_list: List of K values for Top-K accuracy
        
    Returns:
        metrics: Dict of {top{k}_acc: float}
    """
    B, K, L = pred_patterns.shape
    M = ref_bank.shape[0]
    
    # Flatten Batch and K for batched matrix multiplication
    # [B*K, L]
    preds_flat = pred_patterns.reshape(B*K, L)
    targets_flat = target_ids.reshape(B*K)
    
    # Filter out padding targets (ID = -1)
    mask = targets_flat != -1
    if mask.sum() == 0:
        return {f"top{k}_acc": 0.0 for k in topk_list}
        
    valid_preds = preds_flat[mask]   # [N_valid, L]
    valid_targets = targets_flat[mask] # [N_valid]
    
    # Compute Cosine Similarity
    # Normalize preds and bank for Cosine
    valid_preds_n = torch.nn.functional.normalize(valid_preds, p=2, dim=1)
    ref_bank_n = torch.nn.functional.normalize(ref_bank, p=2, dim=1)
    
    # Sim: [N_valid, M]
    sim_matrix = torch.matmul(valid_preds_n, ref_bank_n.T)
    
    metrics = {}
    max_k = max(topk_list)
    
    # Get Top-K indices: [N_valid, Max_K]
    _, topk_indices = torch.topk(sim_matrix, k=min(max_k, M), dim=1)
    
    # Retrieve Top-K IDs: [N_valid, Max_K]
    topk_pred_ids = ref_ids[topk_indices]
    
    for k in topk_list:
        if k > M:
            metrics[f"top{k}_acc"] = 1.0 # Trivial if K > M
            continue
            
        # Check if target is in top K columns
        # [N_valid, k]
        current_preds = topk_pred_ids[:, :k]
        
        # [N_valid, 1]
        t = valid_targets.unsqueeze(1)
        
        # Hit: [N_valid]
        hits = (current_preds == t).any(dim=1).float()
        
        metrics[f"top{k}_acc"] = hits.mean().item()
        
    return metrics

def visualize_separation(
    model,
    dataset,
    device,
    save_dir,
    num_samples=20,
    crystal_db_path=None,
    perfect_only=False,
    run_timestamp="",
):
    """Visualizes separation results."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Establish DB connection if available
    db_conn = None
    if crystal_db_path and HAS_CRYSTAL_LIBS:
        if os.path.exists(crystal_db_path):
            try:
                db_conn = connect(crystal_db_path)
                print(f"Connected to Crystal DB: {crystal_db_path}")
            except Exception as e:
                print(f"Failed to connect to Crystal DB: {e}")
        else:
             print(f"Crystal DB path not found: {crystal_db_path}")

    # Build Reference Bank for Identification Visualization
    from src.data.rruff_dataset import RRUFFOnlineMixingDataset
    if isinstance(dataset, RRUFFOnlineMixingDataset):
        ref_bank_np = np.stack([p[2] for p in dataset.phases])
        ref_ids_np = np.array([p[0] for p in dataset.phases], dtype=np.int64)
        ref_bank = torch.from_numpy(ref_bank_np).to(device)
        ref_ids = torch.from_numpy(ref_ids_np).to(device)
    else:
        ref_bank, ref_ids = build_reference_bank(dataset, device)

    # Ensure num_samples doesn't exceed dataset size
    num_samples = min(num_samples, len(dataset))
    if not perfect_only:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        print("Searching for perfect matches (k=2,3,4)...")
        indices = []
        k_coverage = {2:0, 3:0, 4:0}
        perm = np.random.permutation(len(dataset))
        for p_idx in perm:
            sample = dataset[p_idx]
            targets = sample['single_xrds'].unsqueeze(0).to(device)
            mix = sample['multiphase_xrd'].unsqueeze(0).to(device)
            target_energy = (targets ** 2).sum(dim=-1)
            target_is_active = (target_energy > 1e-4).float()
            gt_k = int(target_is_active.sum().item())
            
            if gt_k not in [2, 3, 4]:
                continue
                
            # If we already have 7 of this k (to get total 20 roughly balanced)
            if k_coverage[gt_k] >= 7 and sum(k_coverage.values()) < 19:
                continue
                
            with torch.no_grad():
                preds, activity_logits = model(mix.unsqueeze(1))
                # Soft check id match (simple correlation)
                sep_loss, best_perms = calculate_pit_loss(preds, targets)
                perm_idx = best_perms[0].cpu().numpy()
                inv_perm = np.argsort(perm_idx)
                pred_ordered = preds[0][inv_perm]
                
                # Check correlations mapping targets directly
                from src.utils.metrics import calculate_pearson_correlation
                good_match = True
                for c in range(4): # K=4
                    if target_is_active[0, c] > 0.5:
                        corr = calculate_pearson_correlation(pred_ordered[c].unsqueeze(0), targets[0, c].unsqueeze(0))
                        if hasattr(corr, "item"):
                            corr = corr.item()
                        if corr < 0.95: # Not perfectly matched ID or shape
                            good_match = False
                            break
                        # Check "weight" match by scale
                        t_energy = target_energy[0, c].item()
                        p_energy = (pred_ordered[c]**2).sum().item()
                        # Allow 20% relative energy diff
                        if p_energy < 1e-5 or abs(t_energy - p_energy) / t_energy > 0.2:
                            good_match = False
                            break
                    else:
                        # Should be inactive
                        if (pred_ordered[c]**2).sum().item() > 0.05 * target_energy.max().item():
                            good_match = False
                            break
                
            if good_match:
                indices.append(p_idx)
                k_coverage[gt_k] += 1
                print(f"Found match: {len(indices)}/20 (k={gt_k})")
                if len(indices) >= num_samples:
                    break
        print(f"Found {len(indices)} perfect matches. Required {num_samples}.")

    # 2-Theta Range Configuration (Hardcoded for now as per request)
    THETA_MIN = 10
    THETA_MAX = 80
    
    for i, idx in enumerate(indices):
        # 5*1 subplot arrangement
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True, gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1]})
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        elif not isinstance(axes, np.ndarray):
            axes = [axes]
        
        with torch.no_grad():
            sample = dataset[idx]
            mix = sample['multiphase_xrd'].unsqueeze(0).to(device)
            targets = sample['single_xrds'].unsqueeze(0).to(device)
            
            # Forward pass using the Film model structure
            # Note: The return signature in hybrid_demucs_film.py is:
            # return out, activity_logits
            # which matches the base model signature used here.
            preds, activity_logits = model(mix.unsqueeze(1))
            
            # Calculate PIT and get best permutation
            # We use pairwise_sisdr to show metrics on plot
            sep_loss, best_perms = calculate_pit_loss(preds, targets)
            
            # best_perms maps Pred Index -> Target Index
            # i.e. best_perms[i] = j means Pred[i] matches Target[j]
            perm = best_perms[0].cpu().numpy()
            
            # To reorder Preds to match Targets (Target 0, Target 1, ...),
            # we need to find which Pred matches Target 0, which matches Target 1, etc.
            # This is the inverse permutation.
            inv_perm = np.argsort(perm)
            
            # Reorder preds to match targets
            pred_ordered = preds[0][inv_perm]
            
            # Activity probs
            act_probs = torch.sigmoid(activity_logits)[0].cpu().numpy()
            act_probs_ordered = act_probs[inv_perm]
            
            # Match Preds to Reference Bank
            # pred_ordered: [K, L] tensor
            pred_norm = torch.nn.functional.normalize(pred_ordered, p=2, dim=1)
            ref_norm = torch.nn.functional.normalize(ref_bank, p=2, dim=1)
            
            # Similarity: [K, N_refs]
            sim_matrix = torch.matmul(pred_norm, ref_norm.T)
            
            # Get Top-1 Match
            top1_vals, top1_indices = torch.max(sim_matrix, dim=1)
            pred_ids = ref_ids[top1_indices].cpu().numpy()
            top1_scores = top1_vals.cpu().numpy()
            # ---------------------------

            targets_np = targets[0].cpu().numpy()
            mix_np = mix[0].cpu().numpy()
            pred_ordered_np = pred_ordered.cpu().numpy()
            
            # Get Phase IDs
            phase_ids = sample['phase_ids'].numpy()
            
            # Calculate metrics for each pair for visualization
            from src.utils.metrics import calculate_pearson_correlation, calculate_sisdr
            pair_metrics = []
            for k in range(len(targets_np)):
                p = torch.from_numpy(pred_ordered_np[k]).float()
                t = torch.from_numpy(targets_np[k]).float()
                corr = calculate_pearson_correlation(p, t)
                sdr = calculate_sisdr(p, t)
                pair_metrics.append((corr, sdr))
            
            # Plot 1: Mixture
            ax_mix = axes[0]
            ax_mix.plot(mix_np, label='Input Mixture', color='black', alpha=0.6, linewidth=2)
            ax_mix.plot(pred_ordered_np.sum(axis=0), label='Sum of Preds', color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax_mix.set_title(f"Sample {idx}: Mixture Reconstruction\nPerm (P->T): {perm}")
            ax_mix.legend(loc='upper right', fontsize='small')
            ax_mix.grid(True, alpha=0.3)
            ax_mix.set_ylabel("Intensity")
            
            # Plot Components (Assume 4 components)
            cmap = plt.get_cmap("tab10")
            
            for k in range(len(targets_np)):
                if k + 1 >= len(axes):
                    break
                
                ax = axes[k + 1]
                color = cmap(k % 10)
                
                # Get ID
                pid = int(phase_ids[k])
                id_str = f"ID:{pid}" if pid != -1 else "Pad"
                
                # GT Plot
                is_gt_active = targets_np[k].max() > 1e-4
                if is_gt_active:
                    ax.fill_between(range(len(targets_np[k])), targets_np[k], color=color, alpha=0.2, label=f'GT {k} ({id_str})')
                    ax.plot(targets_np[k], color=color, alpha=0.6, linewidth=1.5)
                else:
                    ax.text(0.5, 0.5, f"GT {k} (Silent) - {id_str}", transform=ax.transAxes, ha='center', va='center', fontsize=10, color=color, alpha=0.6)
                
                # Pred Plot
                is_pred_active = act_probs_ordered[k] > 0.5
                linestyle = '-' if is_pred_active else ':'
                linewidth = 2.0 if is_pred_active else 1.5
                
                corr, sdr = pair_metrics[k]
                label_pred = f'Pred {k} (r={corr:.2f}, SI-SDR={sdr:.2f})'
                
                # Retrieve Predicted ID and Score
                pred_id = pred_ids[k]
                pred_conf = top1_scores[k]
                pred_id_str = f"PredID:{pred_id}"
                if not is_pred_active:
                    pred_id_str += "(Inactive)"
                
                # Update label
                label_pred = f'Pred {k} (r={corr:.2f}, SI-SDR={sdr:.2f})\n{pred_id_str} (Cos={pred_conf:.2f})'
                
                # Zero out visualization if channel is predicted as inactive to make plot cleaner
                if not is_pred_active:
                    # Optional: you can plot a flat line or keep the noisy prediction with dashed line
                    # For total clarity, we force it to flat zero visually if it's inactive
                    viz_pred = np.zeros_like(pred_ordered_np[k])
                else:
                    viz_pred = pred_ordered_np[k]
                    
                ax.plot(viz_pred, color=color, linestyle=linestyle, linewidth=linewidth, label=label_pred)

                # =========================================================================
                # Theoretical Pattern Plotting
                # =========================================================================
                if db_conn and pid != -1:
                    try:
                        # Fetch atoms and calculate XRD
                        atoms = db_conn.get_atoms(id=pid)
                        structure = AseAtomsAdaptor.get_structure(atoms)
                        
                        calc = XRDCalculator(wavelength="CuKa", symprec=1e-3)
                        # Use wider range to ensure we cover everything, user specified 10-80
                        pattern = calc.get_pattern(structure, two_theta_range=(THETA_MIN, THETA_MAX))
                        
                        # Convert 2-theta (deg) to Index (0..L-1)
                        # Assume linear mapping: L points cover [THETA_MIN, THETA_MAX]
                        # idx = (deg - min) * (L / (max - min))
                        L = len(mix_np)
                        range_deg = THETA_MAX - THETA_MIN
                        
                        peak_indices = (pattern.x - THETA_MIN) * (L / range_deg)
                        peak_intensities = pattern.y / 100.0 # Scale 0-100 to 0-1
                        
                        # Scale by predicted intensity (max amplitude)
                        scale_factor = pred_ordered_np[k].max() if pred_ordered_np[k].max() > 1e-6 else 1.0
                        peak_intensities = peak_intensities * scale_factor

                        # Filter peaks within range
                        mask = (peak_indices >= 0) & (peak_indices < L)
                        idx_toplot = peak_indices[mask]
                        int_toplot = peak_intensities[mask]
                        
                        # Plot Sticks
                        if len(idx_toplot) > 0:
                            ax.vlines(idx_toplot, 0, int_toplot, colors='blue', linestyles='solid', 
                                      linewidth=1.0, alpha=0.9, label='Theoretical (pymatgen)')
                            
                    except Exception as e:
                        print(f"Error plotting theo pattern for ID {pid}: {e}")
                # =========================================================================
                
                ax.set_ylabel(f"Comp {k}")
                # Display Pred ID info in text
                ax.text(0.02, 0.85, f"{pred_id_str}\nCos={pred_conf:.2f}", transform=ax.transAxes, 
                        fontsize=9, color='darkred', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        
                ax.legend(loc='upper right', fontsize='x-small')
                ax.grid(True, alpha=0.3)

            # Hide unused axes
            for k in range(len(targets_np) + 1, len(axes)):
                axes[k].axis('off')
            
        plt.tight_layout()
        suffix = f"_{run_timestamp}" if run_timestamp else ""
        plt.savefig(os.path.join(save_dir, f"test_sample_{i}_{idx}{suffix}.png"), dpi=150)
        plt.close()

def evaluate_model(model, test_loader, device, limit_batches=None, args=None):
    model.eval()
    
    # Build Reference Bank for Identification
    from src.data.rruff_dataset import RRUFFOnlineMixingDataset
    if isinstance(test_loader.dataset, RRUFFOnlineMixingDataset):
        # Build reference strictly from the loaded RRUFF phases
        ref_bank_np = np.stack([p[2] for p in test_loader.dataset.phases])
        ref_ids_np = np.array([p[0] for p in test_loader.dataset.phases], dtype=np.int64)
        ref_bank = torch.from_numpy(ref_bank_np).to(device)
        ref_ids = torch.from_numpy(ref_ids_np).to(device)
    else:
        ref_bank, ref_ids = build_reference_bank(test_loader.dataset, device)
    
    accumulated = {
        'loss': 0.0,
        'si_sdr': 0.0,
        'pearson_corr': 0.0,
        'sir': 0.0,
        'sar': 0.0,
        'delta_2theta': 0.0,
        'fwhm_error': 0.0,
    }

    id_acc_accum = {
        f'id_acc_top{k}': 0.0 for k in range(1, 11)
    }
    
    accumulated.update(id_acc_accum)
    
    steps = 0
    
    with torch.no_grad():
        # Pre-normalize for cosine similarity
        ref_norm = torch.nn.functional.normalize(ref_bank, p=2, dim=1) # [N_refs, L]

        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            if limit_batches is not None and i >= limit_batches:
                break
            mix = batch['multiphase_xrd'].to(device)
            targets = batch['single_xrds'].to(device)
            phase_ids = batch['phase_ids'].to(device) # [B, K]
            
            # Forward
            preds, activity_logits = model(mix.unsqueeze(1))
            
            
            # Ensure target K matches pred K exactly for pure 2,3 tests
            if targets.shape[1] < preds.shape[1]:
                diff = preds.shape[1] - targets.shape[1]
                # Pad targets and phase_ids with silent channels
                targets = torch.cat([targets, torch.zeros(targets.shape[0], diff, targets.shape[2], device=device)], dim=1)
                phase_ids = torch.cat([
                    phase_ids,
                    torch.full((phase_ids.shape[0], diff), -1, device=device, dtype=torch.long)
                ], dim=1)
                
            
            # Loss & Permutation
            sep_loss, best_perms = calculate_pit_loss(preds, targets)
            
            # Align Predictions to Targets
            B, K, L = preds.shape
            
            inv_perms = torch.argsort(best_perms, dim=1) # [B, K]
            
            # Align preds to match targets
            aligned_preds = torch.gather(preds, 1, inv_perms.unsqueeze(-1).expand(-1, -1, L))
            
            # Reranked Top-K identification
            pred_flat_unaligned = preds.view(-1, L)
            pred_norm_unaligned = torch.nn.functional.normalize(pred_flat_unaligned, p=2, dim=1)
            
            sim_matrix_unaligned = torch.matmul(pred_norm_unaligned, ref_norm.T) # [B*K, N_refs]
            top20_vals, top20_indices = torch.topk(sim_matrix_unaligned, k=20, dim=1) # [B*K, 10]
            
            final_top20_ranked_ids = []

            for i in range(B * K):
                if args.disable_rerank:
                    final_top20_ranked_ids.append(top20_indices[i])
                    continue
                pred_sig = pred_flat_unaligned[i]
                cand_idx = top20_indices[i] # [20]
                cand_sigs = ref_bank[cand_idx] # [10, L]
                
                pred_diff = torch.diff(pred_sig)
                pred_diff = torch.cat([pred_diff, torch.zeros(1, device=device)])
                pred_diff_norm = torch.nn.functional.normalize(pred_diff.unsqueeze(0), p=2, dim=1) # [1, L]
                
                cand_diffs = torch.diff(cand_sigs, dim=1)
                cand_diffs = torch.cat([cand_diffs, torch.zeros(20, 1, device=device)], dim=1)
                cand_diffs_norm = torch.nn.functional.normalize(cand_diffs, p=2, dim=1) # [10, L]
                
                deriv_sims = torch.matmul(pred_diff_norm, cand_diffs_norm.T).squeeze(0) # [20]
                
                peak_penalties = torch.ones(20, device=device)
                
                for j in range(20):
                    cand_sig = cand_sigs[j]
                    if cand_sig.max() < 1e-4:
                        continue
                        
                    top_peak_vals, top_peak_idx = torch.topk(cand_sig, k=min(3, L))
                    
                    margin = args.margin
                    valid_peaks_found = 0
                    
                    for p_idx in top_peak_idx:
                        if cand_sig[p_idx] < cand_sig.max() * 0.1: # Skip negligible peaks
                            valid_peaks_found += 1
                            continue
                            
                        start_idx = max(0, p_idx.item() - margin)
                        end_idx = min(L, p_idx.item() + margin + 1)
                        
                        window_max = pred_sig[start_idx:end_idx].max()
                        if window_max > 0.05 * pred_sig.max():
                            valid_peaks_found += 1
                            
                    threshold_peaks = sum((cand_sig[top_peak_idx] >= cand_sig.max() * 0.1).tolist())
                    if valid_peaks_found < min(2, threshold_peaks):
                        peak_penalties[j] = 0.0
                
                c_sim = top20_vals[i] # [20]
                final_scores = (args.alpha * c_sim + (1.0 - args.alpha) * deriv_sims) * peak_penalties
                
                sorted_scores, sorted_idx = torch.sort(final_scores, descending=True)
                reranked_top20_ids = cand_idx[sorted_idx] # [20]
                final_top20_ranked_ids.append(reranked_top20_ids)

            reranked_top20_tensor = torch.stack(final_top20_ranked_ids, dim=0)
            pred_topk_ids = ref_ids[reranked_top20_tensor] # [B*K, 10]

            
            # True activity
            target_energy = (targets ** 2).sum(dim=-1) # [B, K]
            target_is_active = (target_energy > 1e-4).float() # [B, K]
            
            aligned_pred_topk_ids = torch.gather(
                pred_topk_ids.view(B, K, 20),
                1,
                inv_perms.unsqueeze(-1).expand(-1, -1, 20)
            ) # [B, K, 10]

            aligned_pred_topk_ids_flat = aligned_pred_topk_ids.view(-1, 20) # [B*K, 10]
            
            # We must only score active targets
            gt_ids_flat = phase_ids.view(-1, 1) # [B*K, 1]
            target_is_active_flat = target_is_active.view(-1)
            
            is_match = (aligned_pred_topk_ids_flat == gt_ids_flat) # [B*K, 10]
            cum_match = torch.cumsum(is_match.float(), dim=1) # [B*K, 10]
            
            total_active = target_is_active_flat.sum().item()
            
            if total_active > 0:
                for k in range(10):
                    any_match_in_top_k = (cum_match[:, k] > 0).float()
                    acc_k = (any_match_in_top_k * target_is_active_flat).sum().item() / total_active
                    accumulated[f'id_acc_top{k+1}'] += acc_k
            else:
                 for k in range(10):
                    accumulated[f'id_acc_top{k+1}'] += 1.0
            
            batch_metrics = calculate_separation_metrics(
                pred_patterns=aligned_preds,
                target_patterns=targets,
                two_theta_range=(5.0, 90.0), 
                calc_detailed=True
            )
            
            accumulated['loss'] += sep_loss.item()
            accumulated['si_sdr'] += batch_metrics['si_sdr']
            accumulated['pearson_corr'] += batch_metrics['pearson_corr']
            accumulated['sir'] += batch_metrics['sir']
            accumulated['sar'] += batch_metrics['sar']
            accumulated['delta_2theta'] += batch_metrics['delta_2theta']
            accumulated['fwhm_error'] += batch_metrics['fwhm_error']
            
            steps += 1

    return {k: v / max(1, steps) for k, v in accumulated.items()}


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=os.environ.get("PATH_DATA_SINGLEPHASE"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("PATH_OUTPUT_XDECOMPOSER_EVAL", "test_results/xdecomposer_eval"),
    )
    parser.add_argument("--quick", action="store_true", help="Run only 5 batches for debugging/preview")
    parser.add_argument("--num_vis", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument('--perfect_only', action='store_true', help='Only visualize perfect matches')
    parser.add_argument(
        "--crystal_db",
        type=str,
        default=os.environ.get("PATH_DATA_CRYSTAL_DB"),
        help="Path to Crystal DB for theoretical pattern generation",
    )
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for Cosine Similarity')
    parser.add_argument('--margin', type=int, default=5, help='Peak shift tolerance')
    parser.add_argument('--hard_threshold', type=float, default=0.5, help='Hard threshold for final score')
    parser.add_argument('--disable_rerank', action='store_true', help='Disable Rerank Logic and use raw cosine similarity')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--min_k', type=int, default=2, help='Minimum number of phases to mix')
    parser.add_argument('--max_k', type=int, default=None, help='Maximum number of phases to mix (None uses config num_phases)')
    parser.add_argument('--k_weights', type=float, nargs='+', default=None, help='Weights for choosing number of phases (len must be max_k-min_k+1)')
    args = parser.parse_args()
    run_timestamp = extract_timestamp(args.save_dir) or current_timestamp()
    
    # Setup Logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, f"evaluation_{run_timestamp}.log")
    sys.stdout = Logger(log_file)
    print(f"Logging to {log_file}")
    print(f"Run timestamp: {run_timestamp}")
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt['config']
    
    # Override data dir if provided
    if args.data_dir:
        config['singlephase_xrd_db'] = args.data_dir
        
    print(f"Configuration: {config}")
    
    # Load MAE Config
    mae_ckpt_path = (
        os.environ.get("PATH_CKPT_PRETRAIN")
        or config.get("mae_checkpoint", "")
    )
    if not mae_ckpt_path:
        raise ValueError("Missing encoder checkpoint path.")
    print(f"Loading MAE config from {mae_ckpt_path}")
    mae_ckpt = torch.load(mae_ckpt_path, map_location='cpu')
    mae_conf = mae_ckpt.get('config', {})
    
    # Build Models
    mae = XRDMaskedAutoencoder(
        xrd_length=config['xrd_length'],
        d_model=mae_conf.get('d_model', 768),
        n_layers=mae_conf.get('n_layers', 4),
        n_heads=mae_conf.get('n_heads', 12),
        decoder_d_model=mae_conf.get('decoder_dim', 512),
        decoder_n_layers=mae_conf.get('decoder_layers', 4)
    )
    
    model = build_xdecomposer(
        mae,
        num_sources=config['num_phases'],
        cnn_channels=config.get('cnn_channels', [64, 128, 256, 512]),
        cnn_kernels=config.get('cnn_kernels', None),
        cnn_strides=config.get('cnn_strides', None),
        use_transformer=not config.get('no_transformer', False),
        use_film=not config.get('no_film', False),
        use_skip_connections=not config.get('no_skip_connections', False),
        mask_type=config.get('mask_type', 'soft')
    )
    
    # Load State Dict
    # Handle DDP state dict (strip 'module.' prefix)
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA is not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Data Loader
    test_max_k = args.max_k if args.max_k is not None else config['num_phases']
    kwargs = {
        'XRD_LENGTH': config['xrd_length'],
        'MIN_K': args.min_k,
        'MAX_K': test_max_k,
        'AUGMENT': False 
    }
    if getattr(args, 'k_weights', None):
        kwargs['K_WEIGHTS'] = tuple(args.k_weights)
        kwargs['K_DISTRIBUTION'] = 'weighted'
        
    data_config = OnlineMixingConfig(**kwargs)
    
    if args.data_dir and args.data_dir.endswith('.db'):
        from src.data.rruff_dataset import create_rruff_dataloader
        test_loader = create_rruff_dataloader(
            args.data_dir, batch_size=args.batch_size, min_k=args.min_k, max_k=test_max_k, k_weights=args.k_weights,
            split='test', num_folds=args.num_folds, fold=args.fold, seed=args.seed
        )
    else:
        test_loader = create_online_mixing_dataloader(
            args.data_dir if args.data_dir else config['singlephase_xrd_db'],
            args.crystal_db if args.crystal_db else config.get('crystal_db', ""),
            data_config,
            split=args.split,
            train_ratio=0.0,
            val_ratio=0.0,
            batch_size=args.batch_size,
            distributed=False
        )
    
    # Evaluate
    print("Starting evaluation...")
    limit = 5 if args.quick else None
    metrics = evaluate_model(model, test_loader, device, limit_batches=limit, args=args)
    
    # Save results to file
    import json
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    stamped_metrics_path = os.path.join(args.save_dir, f"test_metrics_{run_timestamp}.json")
    stamped_metrics = dict(metrics)
    stamped_metrics["run_timestamp"] = run_timestamp
    stamped_metrics["checkpoint_path"] = args.checkpoint
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(stamped_metrics_path, 'w') as f:
        json.dump(stamped_metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Stamped metrics saved to {stamped_metrics_path}")
    
    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Visualize
    print("\nGenerating visualizations...")
    
    # Determine Crystal DB Path
    # Priority: CLI Argument > Config > Default
    crystal_db_path = args.crystal_db
    if not crystal_db_path:
        crystal_db_path = config.get('crystal_db', None)
        
    visualize_separation(
        model,
        test_loader.dataset,
        device,
        args.save_dir,
        num_samples=args.num_vis,
        crystal_db_path=crystal_db_path,
        perfect_only=getattr(args, 'perfect_only', False),
        run_timestamp=run_timestamp,
    )
    print(f"Visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
