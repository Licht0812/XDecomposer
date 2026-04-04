"""
Evaluation script for DARA on the benchmark dataset.
Runs DARA's phase search and refinement, aligns outputs, and calculates metrics.
"""

import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import logging

from dara.benchmark_loader import OnlineMixingXRDDataset, OnlineMixingConfig, get_splits_from_db
from dara.metrics import calculate_all_metrics
from dara.search.core import search_phases
from dara.search.cosine_matcher import CosineMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def align_dara_output(
    refinement_result,
    ground_truth_ids: torch.Tensor,
    xrd_length: int,
    two_theta_range: Tuple[float, float],
    matcher: CosineMatcher
) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, float]]]]:
    """Aligns DARA's identified phases with ground truth IDs and performs matching."""
    K = ground_truth_ids.shape[0]
    aligned_preds = torch.zeros((K, xrd_length))
    matched_ids_with_scores = [[] for _ in range(K)]
    
    # DARA's structs are stored in plot_data.structs as a dictionary of phase_name -> y_values
    structs = refinement_result.plot_data.structs
    
    # Map each identified phase to its ID
    id_to_pred_y = {}
    id_to_matched_list = {}
    
    for phase_name, y_values in structs.items():
        try:
            # DARA's y_values might have different length, interpolate to xrd_length
            y_tensor = torch.tensor(y_values, dtype=torch.float32)
            
            # Scale back to [0, 1] range because the input .xy was scaled up by 10000
            y_tensor = y_tensor / 10000.0
            
            if len(y_tensor) != xrd_length:
                y_tensor = torch.nn.functional.interpolate(
                    y_tensor.unsqueeze(0).unsqueeze(0), 
                    size=xrd_length, 
                    mode='linear', 
                    align_corners=True
                ).squeeze()
            
            # Use CosineMatcher to get the Top-K IDs for this separated pattern
            matches = matcher.match(y_tensor, topk=10)
            
            # Extract ID from phase_name if possible (e.g., "12345" or "12345.cif")
            import re
            m = re.search(r'(\d+)', str(phase_name))
            if m:
                best_id = int(m.group(1))
                if matches:
                    id_to_matched_list[best_id] = matches
                else:
                    id_to_matched_list[best_id] = [(best_id, 1.0)]
            else:
                if not matches:
                    continue
                best_id = matches[0][0] # The Top-1 ID
                id_to_matched_list[best_id] = matches
                
            id_to_pred_y[best_id] = y_tensor
            
        except Exception as e:
            logger.debug(f"Error matching phase {phase_name}: {e}")
            continue
            
    # Place predicted patterns in the correct slot according to ground truth IDs
    act_logits = torch.full((K,), -10.0) # Low logit for inactive
    
    for i, gt_id in enumerate(ground_truth_ids):
        gt_id_item = gt_id.item()
        if gt_id_item == -1: continue # Padding
        
        # We need to find which predicted phase best matches this GT phase
        # For now, we use the direct ID match if it exists
        if gt_id_item in id_to_pred_y:
            aligned_preds[i] = id_to_pred_y[gt_id_item]
            matched_ids_with_scores[i] = id_to_matched_list[gt_id_item]
            act_logits[i] = 10.0 # High logit for active
        else:
            # If no direct ID match, we might want to pick the most similar one?
            # But the user specifically asked for ID accuracy based on top-k matching.
            pass
            
    return aligned_preds, act_logits, matched_ids_with_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate DARA on UniqCry benchmark dataset")
    parser.add_argument("--db_path", type=str, default="/data/group/project1/Crystal/UniqCryLabeled.db", help="Path to UniqCryLabeled.db")
    parser.add_argument("--npz_dir", type=str, default="/data/group/project1/Crystal/UniqCry/", help="Directory containing single-phase .npz files")
    parser.add_argument("--cif_dir", type=str, default="/data/home/zdhs0019/Projects/xrd_baselines/dara/dataset/uniqcry_cifs", help="Directory containing generated CIF files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of test samples to evaluate (-1 for all)")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Randomly sample a fraction of the test set (e.g. 0.1 for 1/10)")
    parser.add_argument("--instrument", type=str, default="Aeris-fds-Pixcel1d-Medipix3", help="Instrument profile")
    parser.add_argument("--wavelength", type=str, default="Cu", help="Wavelength")
    parser.add_argument("--num_phases", type=int, default=0, help="Specify a fixed number of phases (2, 3, or 4). Default 0 for random mix.")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup Dataset
    config = OnlineMixingConfig()
    if args.num_phases in [2, 3, 4]:
        logger.info(f"Configuring dataset for fixed {args.num_phases} phases.")
        config.MIN_K = args.num_phases
        config.MAX_K = args.num_phases
    else:
        logger.info("Configuring dataset for random 2-4 phases.")

    config.DB_PATH = args.db_path
    config.NPZ_DIR = args.npz_dir
    config.CIF_DIR = args.cif_dir
    
    train_ids, val_ids, test_ids = get_splits_from_db(config.DB_PATH, config)
    all_possible_ids = sorted(train_ids + val_ids + test_ids)
    
    # 1.1 Randomly sample 1/10 of the test set if requested
    eval_ids = test_ids
    if args.sample_ratio < 1.0:
        random.seed(config.SEED)
        num_to_sample = int(len(test_ids) * args.sample_ratio)
        eval_ids = random.sample(test_ids, num_to_sample)
        logger.info(f"Randomly sampled {len(eval_ids)} samples (ratio={args.sample_ratio}) from test set for faster evaluation.")

    dataset = OnlineMixingXRDDataset(
        singlephase_xrd_db_path=config.NPZ_DIR,
        crystal_ids=eval_ids,
        xrd_length=config.XRD_LENGTH,
        min_k=config.MIN_K,
        max_k=config.MAX_K
    )
    
    num_eval = len(dataset) if args.max_samples == -1 else min(len(dataset), args.max_samples)
    logger.info(f"Test split sampled to {len(dataset)} crystal IDs. Will evaluate {num_eval} samples.")
    
    # 1.1 Setup Cosine Matcher with ALL IDs (for top-500 coarse screening)
    matcher = CosineMatcher(
        singlephase_db_path=config.NPZ_DIR,
        crystal_ids=all_possible_ids,
        xrd_length=config.XRD_LENGTH
    )
    
    # Ensure CIFs exist
    cif_files = list(Path(config.CIF_DIR).glob("*.cif"))
    if not cif_files:
        logger.error(f"No CIF files found in {config.CIF_DIR}. Please run scripts/generate_cifs.py first.")
        return
    
    all_metrics = []
    
    # 2. Iterate and Evaluate
    for i in tqdm(range(num_eval), desc="Evaluating DARA"):
        # Save to temp .xy
        temp_xy = os.path.abspath(os.path.join(args.output_dir, f"sample_{i}.xy"))
        item = dataset.save_to_xy(i, temp_xy)
        
        # Ground Truth
        targets = item['single_xrds'] # [K, L]
        gt_ids = item['phase_ids']    # [K]
        multiphase_xrd = item['multiphase_xrd']
        
        try:
            # --- Acceleration: Cosine Similarity Pre-screening ---
            # Get Top 500 candidate IDs from the mixture pattern
            candidate_matches = matcher.match(multiphase_xrd, topk=500)
            candidate_ids = {m[0] for m in candidate_matches}
            
            # Also always include ground truth IDs in candidates for "Oracle" capability if needed
            # but for fair evaluation, we should only use the top-k from cosine similarity.
            # However, if we want to ensure DARA has a chance to find them:
            # candidate_ids.update(gt_ids.tolist()) 
            
            # Filter CIF files to only those in Top 500
            filtered_cifs = [
                f for f in cif_files 
                if int(f.stem) in candidate_ids
            ]
            
            logger.info(f"Sample {i}: Reduced search space from {len(cif_files)} to {len(filtered_cifs)} CIFs using cosine pre-screening.")

            # Search using reduced CIF list
            search_results = search_phases(
                pattern_path=temp_xy,
                phases=filtered_cifs,
                max_phases=dataset.max_k,
                wavelength=args.wavelength,
                instrument_profile=args.instrument,
                express_mode=True,
                rpb_threshold=0.5
            )
            
            if not search_results:
                logger.warning(f"No phases identified for sample {i}")
                continue
                
            # Take the best result
            best_result = search_results[0].refinement_result
            
            # 3. Align and Calculate Metrics
            aligned_preds, act_logits, matched_ids_with_scores = align_dara_output(
                best_result,
                gt_ids,
                dataset.xrd_length,
                dataset.two_theta_range,
                matcher
            )
            
            # Add batch dimension for metric functions
            aligned_preds = aligned_preds.unsqueeze(0)
            targets = targets.unsqueeze(0)
            act_logits = act_logits.unsqueeze(0)
            best_perms = torch.arange(dataset.max_k).unsqueeze(0)
            
            metrics = calculate_all_metrics(
                aligned_preds=aligned_preds,
                targets=targets,
                act_logits=act_logits,
                best_perms=best_perms,
                matched_ids_with_scores=[matched_ids_with_scores], # Batch dimension
                two_theta_range=dataset.two_theta_range,
                gt_ids=gt_ids
            )
            
            all_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error evaluating sample {i}: {e}")
            continue
        finally:
            if os.path.exists(temp_xy):
                os.remove(temp_xy)
                
    # 4. Summarize Results
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        summary = df.mean().to_dict()
        
        logger.info("\n--- Evaluation Summary ---")
        for k, v in summary.items():
            logger.info(f"{k}: {v:.4f}")
            
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        df.to_csv(os.path.join(args.output_dir, "detailed_metrics.csv"), index=False)
    else:
        logger.error("No samples were successfully evaluated.")

if __name__ == "__main__":
    main()
