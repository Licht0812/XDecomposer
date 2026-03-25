import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from model.dataset import ASEDataset
from model.XQueryer import Xmodel
import random

def visualize(model, dataset, device, entries_dict, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Target number of phases to find
    target_phases = [2, 3, 4]
    found_samples = {} # Dictionary to store {num_phases: (index, batch_data)}
    
    print("Searching for samples with 2, 3, and 4 phases...")
    # Search through dataset to find one of each
    search_limit = 2000
    for i in range(min(len(dataset), search_limit)):
        batch = dataset[i]
        num_phases = (batch['gt_ratios'] > 1e-6).sum().item()
        if num_phases in target_phases and num_phases not in found_samples:
            found_samples[num_phases] = (i, batch)
            print(f"  Found {num_phases}-phase sample at index {i}")
        if len(found_samples) == len(target_phases):
            break
            
    two_theta = np.linspace(10, 80, 3500)

    for n_phase in target_phases:
        if n_phase not in found_samples:
            print(f"Warning: Could not find a sample with {n_phase} phases.")
            continue
            
        sample_idx, batch = found_samples[n_phase]
        
        # Prepare input
        intensity = batch['intensity'].unsqueeze(0).to(device)
        element = batch['element'].unsqueeze(0).to(device)
        gt_xrds = batch['gt_xrds'] # MAX_K, 3500
        gt_ratios = batch['gt_ratios'] # MAX_K
        gt_ids = batch['gt_ids'] # MAX_K

        with torch.no_grad():
            outputs = model(intensity, element)
            pred_xrds = outputs['xrds'][0].cpu() # num_slots, 3500
            pred_ratios = outputs['ratios'][0].cpu() # num_slots
            feat_logits = outputs['feat_logits'][0].cpu() # num_slots, num_classes
            
        # Hungarian Matching for visualization alignment
        valid_gt_mask = gt_ratios > 1e-6
        num_valid_gt = valid_gt_mask.sum().item()
        
        # Convert to float32 for cdist
        cost_matrix = torch.cdist(pred_xrds.float(), gt_xrds[valid_gt_mask].float(), p=2).numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_slots = set(row_ind)
        
        # Plotting - Always show all slots
        num_slots = pred_xrds.shape[0]
        num_rows = num_slots + 1
        fig, axes = plt.subplots(num_rows, 1, figsize=(12, 4 * num_rows), sharex=True)
        if num_rows == 1: axes = [axes]
        
        # Plot 1: Mixed XRD
        axes[0].plot(two_theta, batch['intensity'].numpy(), color='black', label='Mixed Input XRD')
        axes[0].set_title(f'Sample {sample_idx} - {n_phase} Phases Mixture (GT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot all slots
        print(f'\nVisualizing {n_phase}-phase sample (Index {sample_idx}):')
        for slot_idx in range(num_slots):
            ax = axes[slot_idx + 1]
            p_ratio = pred_ratios[slot_idx].item()
            
            # Explicit Cosine Similarity Retrieval
            # slot_feat: (feature_dim,)
            # weight_bank: (num_classes, feature_dim)
            slot_feat = outputs['features'][0, slot_idx] 
            weight_bank = model.feat_cls_head.weight # (100315, 256)
            
            # Normalize for Cosine Similarity
            slot_feat_norm = F.normalize(slot_feat, p=2, dim=0)
            weight_bank_norm = F.normalize(weight_bank, p=2, dim=1)
            
            # Compute similarities: (num_classes,)
            cos_sim = torch.matmul(weight_bank_norm, slot_feat_norm)
            
            # Get Top-1
            retrieved_id = torch.argmax(cos_sim).item()
            max_sim = cos_sim[retrieved_id].item()
            
            retrieved_mpid = entries_dict.get(str(retrieved_id), {}).get("mpid", "Unknown")
            
            if slot_idx in matched_slots:
                # This slot is matched to a GT phase
                match_pos = list(row_ind).index(slot_idx)
                gt_idx_in_valid = col_ind[match_pos]
                gt_idx = torch.where(valid_gt_mask)[0][gt_idx_in_valid]
                
                g_ratio = gt_ratios[gt_idx].item()
                g_id = gt_ids[gt_idx].item()
                gt_mpid = entries_dict.get(str(int(g_id)), {}).get("mpid", "Unknown")
                
                ax.plot(two_theta, gt_xrds[gt_idx].numpy(), color='orange', linestyle='--', alpha=0.7, 
                        label=f'Standard XRD (GT ID: {int(g_id)} | {gt_mpid})')
                ax.plot(two_theta, pred_xrds[slot_idx].numpy(), color='royalblue', alpha=0.8, 
                        label=f'Predicted XRD (Slot {slot_idx} | Sim: {max_sim:.3f} | ID: {retrieved_id})')
                
                ax.set_title(f'Slot {slot_idx} [Matched]: Pred Ratio {p_ratio:.2f} | Real Ratio {g_ratio:.2f}')
                print(f'  Slot {slot_idx} [Matched]: GT ID {int(g_id)} ({gt_mpid}) | Pred ID {retrieved_id} ({retrieved_mpid})')
                print(f'           Cosine Sim: {max_sim:.4f} | Pred Ratio {p_ratio:.4f}, GT Ratio {g_ratio:.4f}')
            else:
                # This slot is extra (not matched to any GT)
                ax.plot(two_theta, pred_xrds[slot_idx].numpy(), color='gray', alpha=0.5, 
                        label=f'Predicted XRD (Slot {slot_idx} | Sim: {max_sim:.3f} | ID: {retrieved_id})')
                ax.set_title(f'Slot {slot_idx} [Extra]: Pred Ratio {p_ratio:.2f} | No GT Match')
                print(f'  Slot {slot_idx} [Extra]: Pred ID {retrieved_id} ({retrieved_mpid}) | Sim: {max_sim:.4f} | Pred Ratio {p_ratio:.4f}')
            
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

        plt.xlabel('2-Theta (degrees)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'vis_{n_phase}phase_sample.png')
        plt.savefig(save_path)
        print(f'Saved visualization to {save_path}')
        plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='/data/home/zdhs0019/Projects/xrd_baselines/XQueryer/output/2026-03-24_1725/checkpoints/checkpoint_0010.pth', type=str)
    parser.add_argument('--db_path', default='/data/group/project1/Crystal/UniqCryLabeled.db', type=str)
    parser.add_argument('--npz_dir', default='/data/group/project1/Crystal/UniqCry', type=str)
    parser.add_argument('--entries_dict', default='./src/entries_dict.json', type=str)
    parser.add_argument('--num_slots', default=4, type=int)
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--num_classes', default=100315, type=int)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load entries dictionary
    with open(args.entries_dict, 'r') as f:
        entries_dict = json.load(f)
    print(f"Loaded crystal entries from {args.entries_dict}")

    model = Xmodel(embed_dim=3500, num_slots=args.num_slots, feature_dim=args.feature_dim, num_classes=args.num_classes)
    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    # Use ASEDataset - note that it returns random mixtures on each call
    testset = ASEDataset(args.db_path, args.npz_dir, mode='test', encode_element=True, num_classes=args.num_classes)
    
    visualize(model, testset, device, entries_dict)
