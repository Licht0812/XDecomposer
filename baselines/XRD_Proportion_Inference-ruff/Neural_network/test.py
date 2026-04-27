import os
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from data_utils import get_dataloaders, OnlineMixingConfig
from model import get_model
from metrics_utils import SeparationLoss, calculate_all_metrics

# =============================================================================
# Configuration
# =============================================================================

config = OnlineMixingConfig(
    MIN_K=2,
    MAX_K=4,
    MIN_WEIGHT=0.15,
    XRD_LENGTH=3500,
    AUGMENT=False, # No augment for test
    NOISE_LEVEL=0.01,
    SEED=7
)

DB_PATH = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data'
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_separation_model.pth'
NUM_WORKERS = 8

# =============================================================================
# Evaluation Function
# =============================================================================

def build_reference_library(test_loader):
    """
    Builds a reference library of pure XRD patterns for all IDs in the test set.
    """
    print("Building reference library from test set IDs...")
    library = {}
    
    # Iterate once to collect unique patterns for each ID
    for batch in test_loader:
        xrds = batch['single_xrds'] # [B, K, L]
        pids = batch['phase_ids']   # [B, K]
        weights = batch['weights']  # [B, K]
        
        B, K, L = xrds.shape
        for i in range(B):
            for k in range(K):
                pid = int(pids[i, k])
                w = float(weights[i, k])
                if pid != -1 and w > 1e-6 and pid not in library:
                    # Pure pattern = scaled_xrd / weight
                    pure = xrds[i, k] / w
                    library[pid] = pure.cpu()
                    
    ref_ids = sorted(list(library.keys()))
    ref_patterns = torch.stack([library[rid] for rid in ref_ids])
    
    print(f"Reference library built with {len(ref_ids)} unique crystal types.")
    return ref_patterns.to(DEVICE), ref_ids

def evaluate_model(model, test_loader, ref_lib, ref_ids, num_phases):
    model.eval()
    running_metrics = {
        'rwp': 0.0, 'pearson': 0.0, 'si_sdr': 0.0, 'sir': 0.0, 'sar': 0.0,
        'delta_2theta': 0.0, 'fwhm_error': 0.0, 'intensity_consistency': 0.0,
        'act_f1': 0.0, 'act_precision': 0.0, 'act_recall': 0.0, 'act_exact_match': 0.0,
        'top1_acc': 0.0, 'top3_acc': 0.0, 'top10_acc': 0.0, 'oracle_exact_match': 0.0,
        'quant_mae': 0.0
    }
    count = 0
    
    print(f"Evaluating on test set for {num_phases} phases...")
    since = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['multiphase_xrd'].to(DEVICE)
            targets = batch['single_xrds'].to(DEVICE)
            phase_ids = batch['phase_ids'].to(DEVICE)

            outputs = model(inputs) # [B, 4, L]
            
            # Calculate metrics for the batch
            batch_metrics = calculate_all_metrics(
                outputs, targets, phase_ids, 
                reference_library=ref_lib, 
                reference_ids=ref_ids
            )
            
            for k in running_metrics:
                if k in batch_metrics:
                    running_metrics[k] += batch_metrics[k] * inputs.size(0)
            
            count += inputs.size(0)

    time_elapsed = time.time() - since
    test_metrics = {k: v / count for k, v in running_metrics.items()}
    
    print(f'Evaluation for {num_phases} phases complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f"--- Results for {num_phases} phases ---")
    for k, v in test_metrics.items():
        print(f'{k:20s}: {v:.4f}')
    print("-" * 30)
    
    return test_metrics

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    DB_PATH = '/data/group/project1/Crystal/UniqRruffCrystal.db'
    NUM_FOLDS = 5
    
    all_results = {2: [], 3: [], 4: []}
    
    for num_phases in [2, 3, 4]:
        print(f"\\n{'='*50}")
        print(f"Evaluating for {num_phases} phases")
        print(f"{'='*50}\\n")
        
        config.MIN_K = num_phases
        config.MAX_K = num_phases
        
        for fold in range(NUM_FOLDS):
            print(f"--- Fold {fold} ---")
            _, _, test_loader = get_dataloaders(
                DB_PATH, config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                num_folds=NUM_FOLDS, fold=fold
            )

            ref_lib, ref_ids = build_reference_library(test_loader)

            model = get_model("baseline", out_channels=4).to(DEVICE) # output channels remains 4
            model_path = f'best_separation_model_fold_{fold}.pth'
            
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            else:
                print(f"Warning: {model_path} not found. Using randomly initialized model.")

            metrics = evaluate_model(model, test_loader, ref_lib, ref_ids, num_phases)
            all_results[num_phases].append(metrics)
            
    # Calculate and print averages
    print("\\n\\n" + "="*50)
    print("FINAL AVERAGED RESULTS ACROSS 5 FOLDS")
    print("="*50)
    
    for num_phases in [2, 3, 4]:
        print(f"\\n--- Average Results for {num_phases} phases ---")
        fold_metrics = all_results[num_phases]
        if not fold_metrics: continue
        
        keys = fold_metrics[0].keys()
        avg_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) for k in keys}
        std_metrics = {k: np.std([fm[k] for fm in fold_metrics]) for k in keys}
        
        for k in keys:
            print(f'{k:20s}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}')

