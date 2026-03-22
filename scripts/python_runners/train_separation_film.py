"""Hybrid Demucs-MAE Training Script for XRD Separation."""

import argparse
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import swanlab as sw
import logging
from typing import Dict, Any
from dataclasses import asdict

# Adjust sys.path to include root project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Models
from src.models.xrd_transformer import XRDMaskedAutoencoder
from src.models.hybrid_film import build_hybrid_model

# Data
from src.data.online_mixing_dataset import OnlineMixingXRDDataset, create_online_mixing_dataloader
from src.data.config import OnlineMixingConfig
from src.data.core import XRDCollateFunction

# Utils
from src.utils.distributed import setup_ddp, cleanup_ddp
from src.utils.logging import setup_logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.optimization import get_cosine_schedule_with_warmup, NoamScheduler
from src.utils.metrics import calculate_sisdr, calculate_peak_position_metrics_batch
from src.losses import calculate_pit_loss, calculate_pit_sisdr

import matplotlib.pyplot as plt

def visualize_separation(model, dataset, device, rank, epoch, save_dir, num_samples=2):
    """Visualizes separation results."""
    model.eval()
    if rank != 0: return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            mix = sample['multiphase_xrd'].unsqueeze(0).to(device) # [1, L]
            targets = sample['single_xrds'].unsqueeze(0).to(device) # [1, K, L]
            
            # Predict
            preds, activity_logits = model(mix.unsqueeze(1)) # [1, K, L]
            
            # Find best permutation using PIT logic
            _, best_perms = calculate_pit_loss(preds, targets) # best_perms: [1, K]
            
            perm = best_perms[0].cpu().numpy()
            pred_ordered = preds[0][perm].cpu().numpy() # Reorder prediction to match target
            
            # Get Activity Preds
            act_probs = torch.sigmoid(activity_logits)[0].cpu().numpy()
            
            targets_np = targets[0].cpu().numpy()
            mix_np = mix[0].cpu().numpy()
            
            # Plot 1: Mix vs Sum of Preds
            ax1 = axes[i, 0]
            ax1.plot(mix_np, label='Input Mixture', color='black', alpha=0.5, linewidth=2)
            ax1.plot(pred_ordered.sum(axis=0), label='Sum of Preds', color='red', linestyle='--', alpha=0.8)
            ax1.set_title(f"Sample {idx}: Mixture Reconstruction")
            ax1.legend()
            
            # Plot 2: Individual Components
            ax2 = axes[i, 1]
            # Use Tab10 colormap for distinct colors
            cmap = plt.get_cmap("tab10")
            
            for k in range(len(targets_np)):
                color = cmap(k % 10)
                offset = k * 1.1
                
                # Check if GT is silent
                gt_max = targets_np[k].max()
                is_gt_silent = gt_max < 1e-4
                
                # GT Plot: Filled area with alpha
                if not is_gt_silent:
                    ax2.fill_between(
                        range(len(targets_np[k])), 
                        targets_np[k] + offset, 
                        offset, 
                        color=color, 
                        alpha=0.3, 
                        label=f'GT {k}'
                    )
                    # Add base line for GT
                    ax2.plot(targets_np[k] + offset, color=color, alpha=0.4, linewidth=1)
                else:
                    # Optional: Mark silent GT location
                    ax2.text(0, offset + 0.1, f"GT {k} (Silent)", fontsize=8, color=color, alpha=0.6)

                # Pred Plot: Solid dark line
                # Determine line style based on activity head
                linestyle = '-' if act_probs[k] > 0.5 else ':'
                linewidth = 1.5 if act_probs[k] > 0.5 else 1.0
                act_str = f"P={act_probs[k]:.2f}"
                
                ax2.plot(
                    pred_ordered[k] + offset, 
                    color=color, 
                    linestyle=linestyle, 
                    linewidth=linewidth, 
                    label=f'Pred {k} ({act_str})'
                )
                
                # Add text label for component
                ax2.text(-100, offset + 0.5, f"C{k}", color=color, fontweight='bold', ha='right')
            
            ax2.set_title(f"Sample {idx}: Separated Components (Offset)")
            ax2.legend(loc='upper right', fontsize='small', ncol=2)
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_sep_epoch_{epoch+1}.png"))
    plt.close()

def validate_model(model, val_loader, device, rank, num_phases, lambda_sisdr=0.1):
    model.eval()
    accumulated = {
        'loss': 0.0, 'si_sdr': 0.0,
        'peak_recall': 0.0, 'peak_precision': 0.0, 
        'peak_f1': 0.0, 'peak_mean_shift': 0.0,
        'act_acc': 0.0, 'act_f1': 0.0
    }
    val_steps = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False) if rank == 0 else val_loader
        for batch in pbar:
            mix = batch['multiphase_xrd'].to(device)
            targets = batch['single_xrds'].to(device)
            
            with torch.amp.autocast('cuda'):
                preds, activity_logits = model(mix.unsqueeze(1))
                sep_loss, best_perms = calculate_pit_loss(preds, targets, lambda_sisdr=lambda_sisdr, lambda_geo=1.0, mixture_ref=mix, lambda_mix=1.0)
                
            # Compute metric (PIT SI-SDR)
            sisdr = calculate_pit_sisdr(preds, targets)
            
            # --- Activity Metrics ---
            # 1. Get Labels
            target_energy = (targets ** 2).sum(dim=-1)
            target_is_active = (target_energy > 1e-6).float()
            aligned_activity = torch.gather(target_is_active, 1, best_perms)
            
            # 2. Get Predictions
            act_pred = (torch.sigmoid(activity_logits) > 0.5).float()
            
            # 3. Metrics
            # Accuracy
            acc = (act_pred == aligned_activity).float().mean().item()
            
            # F1 (Channel-wise)
            tp = (act_pred * aligned_activity).sum()
            fp = (act_pred * (1-aligned_activity)).sum()
            fn = ((1-act_pred) * aligned_activity).sum()
            f1 = (2*tp / (2*tp + fp + fn + 1e-8)).item()
            
            
            # Compute Peak Metrics (Batch)
            B, K, L = preds.shape
            preds_flat = preds.reshape(B*K, L)
            targets_flat = targets.reshape(B*K, L)
            peak_metrics = calculate_peak_position_metrics_batch(preds_flat, targets_flat)
            
            accumulated['loss'] += sep_loss.item()
            accumulated['si_sdr'] += sisdr
            accumulated['peak_recall'] += peak_metrics['peak_recall']
            accumulated['peak_precision'] += peak_metrics['peak_precision']
            accumulated['peak_f1'] += peak_metrics['peak_f1']
            accumulated['peak_mean_shift'] += peak_metrics['peak_mean_shift']
            accumulated['act_acc'] += acc
            accumulated['act_f1'] += f1
            
            val_steps += 1
            
    return {k: v / max(1, val_steps) for k, v in accumulated.items()}

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Demucs-MAE Training")
    
    # MAE
    p.add_argument("--mae_checkpoint", type=str, required=True)
    # p.add_argument("--mae_d_model", type=int, default=768) # Will be loaded from checkpoint
    # p.add_argument("--mae_n_layers", type=int, default=4)
    # p.add_argument("--mae_decoder_dim", type=int, default=512)
    # p.add_argument("--mae_decoder_layers", type=int, default=4)
    
    # Data
    p.add_argument("--singlephase_xrd_db", type=str, required=True)
    p.add_argument("--crystal_db", type=str, default="")
    p.add_argument("--xrd_length", type=int, default=3500)
    p.add_argument("--num_phases", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--augment", action="store_true")
    
    # Model Architecture
    p.add_argument("--cnn_channels", type=int, nargs='+', default=[64, 128, 256, 512])
    p.add_argument("--cnn_kernels", type=int, nargs='+', default=[15, 8, 8, 10])
    p.add_argument("--cnn_strides", type=int, nargs='+', default=[1, 2, 2, 5])
    
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", type=str, default="./checkpoints_hybrid")
    p.add_argument("--experiment_name", type=str, default="hybrid_demucs_mae")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lambda_sisdr", type=float, default=0.1)
    p.add_argument("--lambda_activity", type=float, default=2.0, help="Weight for activity detection loss")
    p.add_argument("--lambda_geo", type=float, default=1.0)
    p.add_argument("--lambda_tv", type=float, default=0.0)
    p.add_argument("--lambda_mix", type=float, default=1.0)
    p.add_argument("--activity_threshold", type=float, default=0.8)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume separation training")
    p.add_argument("--finetune", type=str, default=None, help="Path to checkpoint to finetune from (loads weights only, resets epoch/optimizer)")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lr_scheduler", type=str, default="noam", choices=["cosine", "noam"])
    p.add_argument("--noam_factor", type=float, default=1.0)
    
    p.add_argument("--vis_interval", type=int, default=50, help="Visualization interval in epochs")
    
    return p.parse_args()

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. Data Config
    config = OnlineMixingConfig(
        XRD_LENGTH=args.xrd_length,
        MIN_K=2, # Allow varying number of phases (2 to args.num_phases)
        MAX_K=args.num_phases,
        AUGMENT=args.augment
    )
    
    if rank == 0:
        setup_logger(args.save_dir, rank)
        
        # Merge args and data config for logging
        swan_config = vars(args).copy()
        swan_config.update(asdict(config))
        
        sw.init(project="Hybrid-Demucs-MAE", experiment_name=args.experiment_name, config=swan_config)
        
    train_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db, args.crystal_db, config, split='train', batch_size=args.batch_size, distributed=True
    )
    val_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db, args.crystal_db, config, split='val', batch_size=args.batch_size, distributed=True
    )
    
    # 2. Model
    # Load Pre-trained MAE Config from Checkpoint
    if rank == 0:
        logging.info(f"Loading MAE config from {args.mae_checkpoint}")
    
    # We load to CPU first to get config
    mae_ckpt = torch.load(args.mae_checkpoint, map_location='cpu')
    mae_config = mae_ckpt.get('config', {})
    
    # Extract MAE params (fallback to defaults if missing in older ckpts)
    # Note: Keys in mae_config depend on train_pretrain.py args
    mae_d_model = mae_config.get('d_model', 768)
    mae_n_layers = mae_config.get('n_layers', 4)
    mae_n_heads = mae_config.get('n_heads', 12)
    mae_decoder_dim = mae_config.get('decoder_dim', 512)
    mae_decoder_layers = mae_config.get('decoder_layers', 4)
    
    if rank == 0:
        logging.info(f"MAE Config detected: d_model={mae_d_model}, n_layers={mae_n_layers}, dec_dim={mae_decoder_dim}")

    # Init placeholder MAE
    mae = XRDMaskedAutoencoder(
        xrd_length=args.xrd_length, 
        d_model=mae_d_model, 
        n_layers=mae_n_layers, 
        n_heads=mae_n_heads,
        decoder_d_model=mae_decoder_dim,
        decoder_n_layers=mae_decoder_layers
    ) # Config matches checkpoint!
    
    # Load Weights into MAE
    if 'model_state_dict' in mae_ckpt:
        mae.load_state_dict(mae_ckpt['model_state_dict'])
    else:
        mae.load_state_dict(mae_ckpt)
        
    # Free memory
    del mae_ckpt
    
    # Build Hybrid
    model = build_hybrid_model(
        mae, 
        num_sources=args.num_phases,
        cnn_channels=args.cnn_channels,
        cnn_kernels=args.cnn_kernels,
        cnn_strides=args.cnn_strides
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, args.warmup_epochs * steps_per_epoch)
    
    if args.lr_scheduler == "noam":
        scheduler = NoamScheduler(optimizer, d_model=mae_d_model, warmup_steps=warmup_steps, factor=args.noam_factor)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps_per_epoch * args.epochs)
    
    # Load Weights or Resume
    start_epoch = 0
    best_loss = float('inf')  # Init best_loss
    if args.resume:
        # Resume Separation Training (Load everything)
        if rank == 0: logging.info(f"Resuming separation training from {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device, weights_only=False)
    elif args.finetune:
        if rank == 0: logging.info(f"Finetuning separation specific model from {args.finetune}")
        _ = load_checkpoint(args.finetune, model, optimizer, scheduler, device, weights_only=True)
        start_epoch = 0
    # else: 
        # MAE weights are already loaded into 'model' via 'build_hybrid_model' which copied 'mae.encoder'
        # So we don't need to load again.
    
    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        
        for batch in pbar:
            mix = batch['multiphase_xrd'].to(device)
            targets = batch['single_xrds'].to(device)
            
            with torch.amp.autocast('cuda'):
                preds, activity_logits = model(mix.unsqueeze(1))
                
                # 1. Separation Loss & Best Permutation
                sep_loss, best_perms = calculate_pit_loss(
                    preds, targets, 
                    alpha=args.alpha, 
                    lambda_sisdr=args.lambda_sisdr,
                    lambda_geo=args.lambda_geo, 
                    mixture_ref=mix,
                    lambda_mix=args.lambda_mix
                )
                
                # 2. Activity Label Construction (Dynamic based on PIT)
                # Calculate target energy to find active sources
                target_energy = (targets ** 2).sum(dim=-1) # [B, K]
                target_is_active = (target_energy > 1e-6).float() # [B, K]
                
                # Reorder target activity to match predictions using best_perms
                # best_perms: [B, K]
                aligned_activity = torch.gather(target_is_active, 1, best_perms) # [B, K]
                
                # 3. Activity Loss
                activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(activity_logits, aligned_activity)
                
                # --- Component vs Prototype Alignment Loss ---
                # Reorder targets to match predictions
                inv_perms = torch.argsort(best_perms, dim=1)
                aligned_targets = torch.gather(targets, 1, inv_perms.unsqueeze(-1).expand(-1, -1, preds.shape[-1]))
                
                # Filter active
                active_mask = (target_is_active > 0.5)
                preds_active = preds[active_mask]
                targets_active = aligned_targets[active_mask]
                
                id_loss = 0.0
                if len(preds_active) > 0:
                    model_obj = model.module if hasattr(model, 'module') else model
                    # model_obj.extract_id_embeds is disabled for fine-tuning
                    
                loss = sep_loss + args.lambda_activity * activity_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            steps += 1
            if rank == 0: 
                # Log Activity Accuracy (Threshold 0.5)
                act_pred = (torch.sigmoid(activity_logits) > 0.5).float()
                act_acc = (act_pred == aligned_activity).float().mean().item()
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", act_acc=f"{act_acc:.2f}")
                sw.log({"train/loss_step": loss.item(), "train/act_loss": activity_loss.item(), "lr": optimizer.param_groups[0]['lr']})
            
        val_metrics = validate_model(model, val_loader, device, rank, args.num_phases, lambda_sisdr=args.lambda_sisdr)
        
        if rank == 0:
            logging.info(
                f"Epoch {epoch+1} | Train: {epoch_loss/steps:.5f} | "
                f"Val Loss: {val_metrics['loss']:.5f} | SI-SDR: {val_metrics['si_sdr']:.2f} | "
                f"F1: {val_metrics['peak_f1']:.3f} | Act Acc: {val_metrics['act_acc']:.2f}"
            )
            sw.log({
                "train/loss": epoch_loss/steps, 
                "val/loss": val_metrics['loss'], 
                "val/act_acc": val_metrics['act_acc'],
                "val/act_f1": val_metrics['act_f1'],
                "val/si_sdr": val_metrics['si_sdr'],
                "val/peak_recall": val_metrics['peak_recall'],
                "val/peak_precision": val_metrics['peak_precision'],
                "val/peak_f1": val_metrics['peak_f1'],
                "val/peak_mean_shift": val_metrics['peak_mean_shift'],
                "lr": optimizer.param_groups[0]['lr']
            })
            
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.save_dir, "best.pt"), vars(args), rank, False)
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.save_dir, "latest.pt"), vars(args), rank, False)
            
            # Visualization
            if (epoch + 1) % args.vis_interval == 0:
                # Use training set for visualization to see fitting, or val set? Usually val set is better to check generalization.
                # Let's use val_loader.dataset
                visualize_separation(
                    model, 
                    val_loader.dataset, # Need dataset for random access
                    device, 
                    rank, 
                    epoch, 
                    args.save_dir
                )
            
    cleanup_ddp()

if __name__ == "__main__":
    main()