import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from data_utils import get_dataloaders, OnlineMixingConfig
from model import get_model
from metrics_utils import align_predictions, calculate_cosine_similarity

def build_reference_library(test_loader, device):
    """
    Builds a reference library of pure XRD patterns for all IDs in the test set.
    """
    print("Building reference library for visualization...")
    library = {}
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
    return ref_patterns.to(device), ref_ids

def retrieve_id(pred_xrd, ref_lib, ref_ids):
    """
    Retrieves the most similar crystal ID from the reference library.
    pred_xrd: [1, L]
    """
    if pred_xrd.max() < 1e-4:
        return -1, 0.0
    
    # Calculate cosine similarity: [1, NumRefs]
    similarities = calculate_cosine_similarity(pred_xrd, ref_lib)
    max_sim, max_idx = torch.max(similarities, dim=1)
    
    return ref_ids[max_idx.item()], max_sim.item()

def visualize_samples(model_path='best_separation_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db_path = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data'
    
    config = OnlineMixingConfig(
        MIN_K=2, MAX_K=4, MIN_WEIGHT=0.15, XRD_LENGTH=3500,
        AUGMENT=False, NOISE_LEVEL=0.01, SEED=42 # Change seed to get different samples
    )
    
    # 1. Load Data & Build Library
    _, _, test_loader = get_dataloaders(db_path, config, batch_size=32, num_workers=4)
    ref_lib, ref_ids = build_reference_library(test_loader, device)
    
    # 2. Load Model
    model = get_model("baseline", out_channels=config.MAX_K).to(device)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Select samples for 2, 3, 4 phases
    selected_batches = []
    found_ks = set()
    
    print("Searching for 2, 3, 4 phase samples in test set...")
    for batch in test_loader:
        num_phases = batch['num_phases']
        for i in range(len(num_phases)):
            k = int(num_phases[i])
            if k in [2, 3, 4] and k not in found_ks:
                # Store a single-sample batch
                single_batch = {k: v[i:i+1] for k, v in batch.items()}
                selected_batches.append(single_batch)
                found_ks.add(k)
        if len(found_ks) == 3:
            break
            
    # Sort by k: 2, 3, 4
    selected_batches.sort(key=lambda x: int(x['num_phases'][0]))
    
    # 4. Plotting
    num_samples = len(selected_batches)
    fig, axes = plt.subplots(num_samples, 5, figsize=(30, 6 * num_samples))
    
    for row_idx, batch in enumerate(selected_batches):
        k_actual = int(batch['num_phases'][0])
        inputs = batch['multiphase_xrd'].to(device)
        targets = batch['single_xrds'].to(device)
        true_pids = batch['phase_ids'].to(device)
        true_weights = batch['weights'].to(device)
        
        with torch.no_grad():
            outputs = model(inputs) # [1, 4, L]
            best_perms = align_predictions(outputs, targets)
            best_perms_expanded = best_perms.unsqueeze(-1).expand(-1, -1, 3500)
            aligned_outputs = torch.gather(outputs, 1, best_perms_expanded)

        # --- Column 0: Mixed XRD ---
        mix_np = inputs[0, 0].cpu().numpy()
        axes[row_idx, 0].plot(mix_np, color='black', lw=1.5)
        axes[row_idx, 0].set_title(f"Sample {row_idx+1}: Mixed ({k_actual} Phases)", fontsize=14, fontweight='bold')
        axes[row_idx, 0].set_ylabel("Intensity", fontsize=12)
        axes[row_idx, 0].grid(True, alpha=0.3)
        
        # --- Columns 1-4: Single Phase Slots ---
        for col_idx in range(4):
            ax = axes[row_idx, col_idx + 1]
            
            # Ground Truth
            gold_xrd = targets[0, col_idx].cpu().numpy()
            true_id = int(true_pids[0, col_idx])
            true_w = float(true_weights[0, col_idx])
            
            # Prediction
            pred_xrd = aligned_outputs[0, col_idx] # tensor [L]
            pred_xrd_np = pred_xrd.cpu().numpy()
            
            # Retrieve ID from Library using Prediction
            ret_id, sim = retrieve_id(pred_xrd.unsqueeze(0), ref_lib, ref_ids)
            
            # Prediction Weight (Sum of intensity / Sum of mixed intensity)
            pred_w = pred_xrd_np.sum() / mix_np.sum() if mix_np.sum() > 0 else 0
            
            # Define status
            status = "ACTIVE" if true_id != -1 else "EMPTY"

            # Plot
            if status == "ACTIVE":
                ax.plot(gold_xrd, color='blue', alpha=0.5, lw=2, label='Gold')
                
            ax.plot(pred_xrd_np, color='red', alpha=0.7, linestyle='--', lw=1.5, label='Pred')
            
            # Formatting
            title = f"Slot {col_idx} [{status}]\n"
            if status == "ACTIVE":
                title += f"True ID: {true_id} (w={true_w:.2f})\n"
                title += f"Pred ID: {ret_id} (Sim:{sim:.3f})\n"
                title += f"Pred w: {pred_w:.2f}"
            else:
                title += f"Pred ID: {ret_id} (Sim:{sim:.3f})\n"
                title += f"Pred w: {pred_w:.2f}"
                # If similarity is very low, it might be noise
                if sim < 0.5:
                    title += " (Noise?)"
                
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.2)
            if col_idx == 0:
                ax.legend(loc='upper right', fontsize='small')
            
            # Set Y limit
            max_val = max(gold_xrd.max(), pred_xrd_np.max(), 0.05)
            ax.set_ylim(0, max_val * 1.3)

    plt.tight_layout()
    save_path = 'xrd_separation_results.png'
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_samples()
