"""Single-Phase XRD Masked Autoencoder Pre-training (MAE)."""

import argparse
import datetime
import logging
import math
import os
import sys
from typing import Dict, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import swanlab as sw
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm

# Adjust sys.path to include root project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Modules
from src.data.single_phase_mae_dataset import SinglePhaseMaskedDataset
from src.data.config import OnlineMixingConfig
from src.models.xrd_transformer import XRDMaskedAutoencoder
from src.utils.distributed import setup_ddp, cleanup_ddp
from src.utils.logging import setup_logger
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.optimization import get_cosine_schedule_with_warmup, NoamScheduler

# Use non-interactive backend
matplotlib.use('Agg')

def unpatchify(patches: torch.Tensor, stride: int, padding: int, xrd_length: int) -> torch.Tensor:
    """Reconstructs 1D signal from patches."""
    batch_size, num_patches, patch_len = patches.shape
    device = patches.device
    
    total_padded_len = (num_patches - 1) * stride + patch_len
    xrd_reconstructed = torch.zeros(batch_size, total_padded_len, device=device)
    counts = torch.zeros(batch_size, total_padded_len, device=device)
    
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        xrd_reconstructed[:, start_idx:end_idx] += patches[:, i, :]
        counts[:, start_idx:end_idx] += 1
        
    xrd_reconstructed = xrd_reconstructed / (counts + 1e-8)
    
    if padding > 0:
        xrd_reconstructed = xrd_reconstructed[:, padding:-padding]
        
    if xrd_reconstructed.shape[1] > xrd_length:
        xrd_reconstructed = xrd_reconstructed[:, :xrd_length]
    elif xrd_reconstructed.shape[1] < xrd_length:
        xrd_reconstructed = F.pad(xrd_reconstructed, (0, xrd_length - xrd_reconstructed.shape[1]))
        
    return xrd_reconstructed

def get_dataloaders(args, rank: int, world_size: int) -> Tuple[DataLoader, DataLoader]:
    if rank == 0:
        logging.info("Initializing Dataset...")
        
    full_dataset = SinglePhaseMaskedDataset(
        singlephase_xrd_db_path=args.singlephase_db,
        xrd_length=args.xrd_length,
        norm_method=args.norm_method,
        data_fraction=args.data_fraction,
        cache_index=True
    )
    
    if rank == 0:
        logging.info(f"Dataset Size: {len(full_dataset)}")
    
    train_size = int(0.90 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True
    )
    
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, epoch, args, rank, world_size):
    model.train()
    if hasattr(model, 'module'):
        model.module.mask_ratio = args.mask_ratio
    else:
        model.mask_ratio = args.mask_ratio
        
    total_loss = torch.zeros(1, device=device)
    steps = 0
    
    pbar = None
    if rank == 0:
        logging.info(f"Epoch {epoch+1}: Mask Ratio: {args.mask_ratio:.2f}")
        pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}", leave=True, file=sys.stdout)
    
    for imgs in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):
            loss, _, _ = model(imgs)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.detach()
        steps += 1
        
        if rank == 0 and pbar:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            pbar.update(1)
            sw.log({"train/loss_step": loss.item()})
            
    if pbar: pbar.close()
        
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (steps * world_size)
    return avg_loss

def validate(model, val_loader, device, rank, limit_batches=50, compute_full_metrics=False):
    model.eval()
    local_metrics = {'loss': 0.0, 'mse': 0.0, 'count': 0.0}
    full_metric_keys = ['si_sdr', 'peak_position_accuracy']
    
    if compute_full_metrics:
        for k in full_metric_keys: local_metrics[k] = 0.0
    
    with torch.no_grad():
        for i, imgs in enumerate(val_loader):
            if limit_batches and i >= limit_batches: break
            imgs = imgs.to(device)
            m = model.module if hasattr(model, 'module') else model
            imgs_norm = m.normalize_xrd(imgs)
            
            with torch.amp.autocast('cuda'):
                loss, pred, mask = model(imgs) 
            
            # Reconstruction
            patches_gt = m.patchify(imgs_norm)
            mask_u = mask.unsqueeze(-1)
            patches_comp = pred * mask_u + patches_gt * (1 - mask_u)
            recon = unpatchify(patches_comp, m.stride, m.padding, m.xrd_length)
            recon = torch.clamp(recon, min=0.0)
            
            mse = F.mse_loss(recon, imgs_norm).item()
            local_metrics['loss'] += loss.item()
            local_metrics['mse'] += mse
            local_metrics['count'] += 1
            
            if compute_full_metrics:
                pass # Metrics logic placeholder

    # Aggregation
    data_list = [local_metrics['loss'], local_metrics['count'], local_metrics['mse']]
    tensor_metrics = torch.tensor(data_list, device=device)
    dist.all_reduce(tensor_metrics, op=dist.ReduceOp.SUM)
    
    count = tensor_metrics[1].item() or 1e-8
    results = {
        'loss': tensor_metrics[0].item() / count,
        'mse': tensor_metrics[2].item() / count
    }
    return results

def visualize(model, val_loader, device, epoch, output_dir, rank):
    if rank != 0: return
    model.eval()
    try:
        imgs = next(iter(val_loader))
        imgs = imgs.to(device)[:4]
        m = model.module if hasattr(model, 'module') else model
        imgs_norm = m.normalize_xrd(imgs)
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            loss, pred, mask = model(imgs)
            
        patches_gt = m.patchify(imgs_norm)
        mask_u = mask.unsqueeze(-1)
        patches_comp = pred * mask_u + patches_gt * (1 - mask_u)
        recon = unpatchify(patches_comp, m.stride, m.padding, m.xrd_length)
        mask_recon = unpatchify(mask_u.expand(-1,-1,m.patch_len).float(), m.stride, m.padding, m.xrd_length)
        
        recon = torch.clamp(recon, min=0.0)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        if not isinstance(axes, np.ndarray): axes = [axes]
        
        for i, ax in enumerate(axes):
            if i >= len(imgs): break
            orig = imgs_norm[i].cpu().numpy()
            rec = recon[i].cpu().numpy()
            msk = mask_recon[i].cpu().numpy()
            
            ax.fill_between(range(len(orig)), orig.min(), orig.max(), where=(msk > 0.5), color='gray', alpha=0.3, label='Masked')
            ax.plot(orig, label='GT', color='black', alpha=0.6, linewidth=1)
            ax.plot(rec, label='Recon', color='red', linestyle='--', alpha=0.8, linewidth=1)
            if i == 0: ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"vis_epoch_{epoch}.png"))
        plt.close()
    except Exception as e:
        print(f"Vis failed: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="XRD MAE Pre-training")
    
    parser.add_argument("--singlephase_db", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mae_pretrain")
    parser.add_argument("--experiment_name", type=str, default="mae_pretrain")
    
    # Data
    parser.add_argument("--xrd_length", type=int, default=OnlineMixingConfig.XRD_LENGTH)
    parser.add_argument("--norm_method", type=str, default=OnlineMixingConfig.NORM_METHOD)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--ohem_ratio", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=0) # For Noam
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "noam"])
    parser.add_argument("--noam_factor", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights_only", action="store_true")
    
    # Loss
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--lambda_cos", type=float, default=0.5)
    parser.add_argument("--lambda_deriv", type=float, default=0.1)
    
    return parser.parse_args()

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        setup_logger(args.output_dir, rank)
        
        # Merge args and base config defaults for logging
        swan_config = vars(args).copy()
        # Note: Pretraining doesn't use all OnlineMixingConfig params, but we log defaults for reference
        # We manually exclude mixing-specific params to avoid confusion, or just include purely common ones?
        # For simplicity and "Pretraining needs it too" request, we can log relevant ones or just rely on args defaults we just set.
        # But to be explicit in SwanLab about "Config Source":
        # swan_config.update({'config_source': 'OnlineMixingConfig defaults'})
        
        sw.init(project="XRD-MAE-Pretrain", name=f"{args.experiment_name}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", config=swan_config)
        
    train_loader, val_loader = get_dataloaders(args, rank, world_size)
    
    model = XRDMaskedAutoencoder(
        xrd_length=args.xrd_length, patch_len=args.patch_len, stride=args.stride,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_d_model=args.decoder_dim, decoder_n_layers=args.decoder_layers, decoder_n_heads=args.decoder_heads,
        dropout=args.dropout, mask_ratio=args.mask_ratio, ohem_ratio=args.ohem_ratio,
        alpha=args.alpha, lambda_cos=args.lambda_cos, lambda_deriv=args.lambda_deriv
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    
    steps_per_epoch = len(train_loader)
    
    if args.lr_scheduler == "noam":
        # If warmup_steps not explicitly set, calculate from warmup_epochs
        warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, args.warmup_epochs * steps_per_epoch)
        scheduler = NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=warmup_steps, factor=args.noam_factor)
    else:
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Resume
    start_epoch = 0
    if args.resume:
        # Load checkpoint (including scheduler state if available)
        start_epoch = load_checkpoint(
            args.resume, 
            model, 
            optimizer if not args.weights_only else None, 
            scheduler if not args.weights_only else None,
            device, 
            args.weights_only
        )
        if not args.weights_only:
            pass
            
    if rank == 0: logging.info(f"Start training: {args.epochs} epochs")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, epoch, args, rank, world_size)
        
        if rank == 0: sw.log({"train/loss_epoch": avg_loss, "train/lr": optimizer.param_groups[0]['lr']})
        
        val_metrics = validate(model, val_loader, device, rank, limit_batches=50 if (epoch+1)%50!=0 else 100, compute_full_metrics=((epoch+1)%50==0))
        
        if rank == 0:
            logging.info(f"Epoch {epoch+1} | Train: {avg_loss:.5f} | Val: {val_metrics['loss']:.5f}")
            sw.log({f"val/{k}": v for k, v in val_metrics.items()})
            visualize(model, val_loader, device, epoch+1, args.output_dir, rank)
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.output_dir, "best_model.pt"), vars(args), rank, verbose=False)
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.output_dir, "checkpoint_latest.pt"), vars(args), rank, verbose=False)

    cleanup_ddp()

if __name__ == "__main__":
    main()
