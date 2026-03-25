import os
import torch
import torch.nn as nn
import numpy as np
import time
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

def evaluate_model(model, test_loader, ref_lib, ref_ids):
    model.eval()
    running_metrics = {
        'rwp': 0.0, 'pearson': 0.0, 'si_sdr': 0.0,
        'top1_acc': 0.0, 'top3_acc': 0.0, 'top10_acc': 0.0
    }
    count = 0
    
    print("Evaluating on test set...")
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
    
    print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print("-" * 30)
    for k, v in test_metrics.items():
        print(f'{k:20s}: {v:.4f}')
    print("-" * 30)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # 1. Create Dataloader
    _, _, test_loader = get_dataloaders(
        DB_PATH, config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # 2. Build Reference Library (for Retrieval Top-K)
    ref_lib, ref_ids = build_reference_library(test_loader)

    # 3. Load Model
    model = get_model("baseline", out_channels=config.MAX_K).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {MODEL_PATH} not found. Using randomly initialized model.")

    # 4. Evaluate
    evaluate_model(model, test_loader, ref_lib, ref_ids)
