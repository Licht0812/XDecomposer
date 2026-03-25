import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from model.dataset import ASEDataset
from model.XQueryer import Xmodel

def calculate_rwp(pred_pattern: torch.Tensor, target_pattern: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Calculate R-weighted Profile (Rwp)."""
    target_pattern, pred_pattern = torch.clamp(target_pattern, min=0), torch.clamp(pred_pattern, min=0)
    diff_sq = (target_pattern - pred_pattern) ** 2
    numerator = torch.sum(diff_sq, dim=-1)
    denominator = torch.sum(target_pattern ** 2, dim=-1) + epsilon
    return torch.sqrt(numerator / denominator).mean().item()

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if pred.ndim == 1:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    dot_product = torch.sum(pred * target, dim=-1)
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    alpha = dot_product / target_energy
    e_target = alpha.unsqueeze(-1) * target
    e_res = pred - e_target
    signal_energy, noise_energy = torch.sum(e_target ** 2, dim=-1), torch.sum(e_res ** 2, dim=-1) + eps
    ratio = torch.clamp(signal_energy / noise_energy, min=1e-10)
    return (10 * torch.log10(ratio)).mean().item()

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
        'sisdr_sum': 0,
        'ratio_mae_sum': 0,
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
                    sisdr = calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                    ratio_err = torch.abs(pred_ratios[b, r] - gt_ratios[b, gt_idx]).item()
                    
                    metrics['rwp_sum'] += rwp
                    metrics['sisdr_sum'] += sisdr
                    metrics['ratio_mae_sum'] += ratio_err
                    
                    sample_results["phases"].append({
                        "gt_id": target_id,
                        "gt_mpid": entries_dict.get(str(target_id), {}).get("mpid", "Unknown"),
                        "gt_ratio": float(gt_ratios[b, gt_idx].item()),
                        "pred_ratio": float(pred_ratios[b, r].item()),
                        "rwp": rwp,
                        "sisdr": sisdr,
                        "top10_preds": [
                            {
                                "id": int(idx),
                                "mpid": entries_dict.get(str(int(idx)), {}).get("mpid", "Unknown")
                            } for idx in top10_indices
                        ]
                    })

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
        print(f"Avg SI-SDR:      {metrics['sisdr_sum']/metrics['total_phases']:.2f} dB")
        print(f"Avg Ratio MAE:   {metrics['ratio_mae_sum']/metrics['total_phases']*100:.2f}%")
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

    # Use updated ASEDataset which supports ID-based splitting
    testset = ASEDataset(args.db_path, args.npz_dir, mode='test', 
                         encode_element=args.atom_embed, num_classes=args.num_classes)
    
    # Use smaller batch size if on CPU to prevent OOM
    batch_size = args.batch_size if torch.cuda.is_available() else 1
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    run_one_epoch(model, test_loader, device, entries_dict, threshold=args.threshold, limit=args.limit)

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

    args = parser.parse_args()
    main()
    print('THE END')
