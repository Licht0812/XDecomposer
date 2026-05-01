import argparse
import math
import os
from typing import Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment

from model.dataset import RRUFFOnlineMixingDataset
from model.XQueryer import Xmodel
from util.logger import Logger

def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    """Calculate Pearson correlation coefficient."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    pred_mean = pred_pattern.mean(dim=-1, keepdim=True)
    target_mean = target_pattern.mean(dim=-1, keepdim=True)
    pred_centered = pred_pattern - pred_mean
    target_centered = target_pattern - target_mean
    numerator = (pred_centered * target_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum(dim=-1) + 1e-8)
    return (numerator / (pred_std * target_std + 1e-8)).mean().item()

def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if pred.ndim == 1:
        pred, target = pred.unsqueeze(0), target.unsqueeze(0)
    dot_product = torch.sum(pred * target, dim=-1)
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    alpha = (dot_product / target_energy).unsqueeze(-1)
    e_target = alpha * target
    e_res = pred - e_target
    signal_energy = torch.sum(e_target ** 2, dim=-1)
    noise_energy = torch.sum(e_res ** 2, dim=-1) + eps
    ratio = torch.clamp(signal_energy / noise_energy, min=1e-10)
    return (10 * torch.log10(ratio)).mean().item()

def get_id_acc_topk(logits: torch.Tensor, target_ids: torch.Tensor, k=10):
    """
    Calculate Top-K Accuracy for phase identification.
    In slot-based architecture, we only calculate this for matched slots.
    """
    # logits: (N_matched, num_classes)
    # target_ids: (N_matched,)
    if logits.size(0) == 0: return 0.0
    _, topk_indices = torch.topk(logits, k=k, dim=1)
    correct = (topk_indices == target_ids.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()

def weighted_xrd_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 10.0,
                      lambda_cos: float = 0.2, lambda_geo: float = 0.1,
                      beta: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    focal = 1.0 + alpha * target
    loss_l1 = (torch.abs(pred - target) * focal).mean()

    loss_cos = 1.0 - F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0), dim=-1, eps=eps).mean()

    pred_sqrt = torch.sqrt(torch.clamp(pred, min=eps))
    target_sqrt = torch.sqrt(torch.clamp(target, min=eps))
    grad1 = torch.abs((pred_sqrt[1:] - pred_sqrt[:-1]) - (target_sqrt[1:] - target_sqrt[:-1])).mean()
    grad2 = torch.abs(
        (pred_sqrt[2:] - 2 * pred_sqrt[1:-1] + pred_sqrt[:-2]) -
        (target_sqrt[2:] - 2 * target_sqrt[1:-1] + target_sqrt[:-2])
    ).mean()
    loss_geo = grad1 + beta * grad2

    return loss_l1 + lambda_cos * loss_cos + lambda_geo * loss_geo

def build_matching_cost(pred_xrds: torch.Tensor, gt_xrds: torch.Tensor) -> torch.Tensor:
    cost_matrix = torch.zeros(pred_xrds.size(0), gt_xrds.size(0), device=pred_xrds.device)
    for i in range(pred_xrds.size(0)):
        for j in range(gt_xrds.size(0)):
            cost_matrix[i, j] = weighted_xrd_loss(pred_xrds[i], gt_xrds[j]).detach()
    return cost_matrix

def compute_sample_objective(pred_xrds: torch.Tensor, pred_ratios: torch.Tensor, feat_logits: torch.Tensor,
                             gt_xrds: torch.Tensor, gt_ratios: torch.Tensor, gt_ids: torch.Tensor,
                             mixture: torch.Tensor, criterion_ratio, criterion_cls,
                             lambda_ratio: float = 1.0, lambda_cls: float = 0.1,
                             lambda_mix: float = 1.0, lambda_empty: float = 0.5):
    valid_gt_mask = gt_ratios > 1e-6
    valid_gt_indices = torch.where(valid_gt_mask)[0]
    matched_rows = torch.zeros(pred_xrds.size(0), dtype=torch.bool, device=pred_xrds.device)
    sample_loss = lambda_mix * F.l1_loss(pred_xrds.sum(dim=0), mixture)
    row_ind, col_ind = np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if valid_gt_indices.numel() > 0:
        gt_active_xrds = gt_xrds[valid_gt_indices]
        cost_matrix = build_matching_cost(pred_xrds.float(), gt_active_xrds.float()).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            gt_idx = valid_gt_indices[c]
            sample_loss = sample_loss + weighted_xrd_loss(pred_xrds[r], gt_xrds[gt_idx])
            sample_loss = sample_loss + lambda_ratio * criterion_ratio(pred_ratios[r], gt_ratios[gt_idx])
            sample_loss = sample_loss + lambda_cls * criterion_cls(feat_logits[r].unsqueeze(0), gt_ids[gt_idx].unsqueeze(0))
            matched_rows[r] = True

    if (~matched_rows).any():
        sample_loss = sample_loss + lambda_empty * pred_xrds[~matched_rows].abs().mean()
        sample_loss = sample_loss + lambda_empty * pred_ratios[~matched_rows].mean()

    return sample_loss, row_ind, col_ind, valid_gt_indices

def run_one_epoch(model, dataloader, optimizer, epoch, mode):
    if mode == 'Train':
        model.train()
        desc = 'Training... '
    else:
        model.eval()
        desc = 'Evaluating... '

    epoch_loss = 0
    metrics_sum = {
        'pearson_corr': 0, 'si_sdr': 0, 'id_acc_top10': 0, 'total_count': 0
    }

    if args.progress_bar:
        pbar = tqdm(total=len(dataloader.dataset), desc=desc, unit='data')
    iters = len(dataloader)

    criterion_ratio = torch.nn.L1Loss()
    criterion_cls = torch.nn.CrossEntropyLoss()

    for i, batch in enumerate(dataloader):
        intensity = batch['intensity'].to(device)
        element = batch['element'].to(device)
        gt_xrds = batch['gt_xrds'].to(device)
        gt_ratios = batch['gt_ratios'].to(device)
        gt_ids = batch['gt_ids'].to(device)

        if mode == 'Train':
            adjust_learning_rate_withWarmup(optimizer, epoch + i / iters, args)
            with torch.cuda.amp.autocast():
                outputs = model(intensity, element)
                pred_xrds = outputs['xrds']
                pred_ratios = outputs['ratios']
                feat_logits = outputs['feat_logits']

                total_loss = 0
                batch_size = intensity.size(0)

                for b in range(batch_size):
                    sample_loss, _, _, _ = compute_sample_objective(
                        pred_xrds[b], pred_ratios[b], feat_logits[b],
                        gt_xrds[b], gt_ratios[b], gt_ids[b], intensity[b],
                        criterion_ratio, criterion_cls
                    )
                    total_loss += sample_loss

                loss = total_loss / batch_size

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                outputs = model(intensity, element)
                pred_xrds = outputs['xrds']
                pred_ratios = outputs['ratios']
                feat_logits = outputs['feat_logits']

                total_loss = 0
                batch_size = intensity.size(0)
                for b in range(batch_size):
                    sample_loss, row_ind, col_ind, valid_gt_indices = compute_sample_objective(
                        pred_xrds[b], pred_ratios[b], feat_logits[b],
                        gt_xrds[b], gt_ratios[b], gt_ids[b], intensity[b],
                        criterion_ratio, criterion_cls
                    )
                    total_loss += sample_loss

                    if valid_gt_indices.numel() > 0:
                        matched_logits = []
                        matched_targets = []
                        for r, c in zip(row_ind, col_ind):
                            gt_idx = valid_gt_indices[c]
                            metrics_sum['pearson_corr'] += calculate_pearson_correlation(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['si_sdr'] += calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            matched_logits.append(feat_logits[b, r])
                            matched_targets.append(gt_ids[b, gt_idx])
                            metrics_sum['total_count'] += 1

                        if matched_logits:
                            metrics_sum['id_acc_top10'] += get_id_acc_topk(torch.stack(matched_logits), torch.stack(matched_targets), k=10) * len(matched_logits)

                loss = total_loss / batch_size

        epoch_loss += loss.item()

        if args.progress_bar:
            pbar.update(len(intensity))
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if args.progress_bar:
        pbar.close()

    # Average metrics
    avg_metrics = {k: (v / metrics_sum['total_count'] if metrics_sum['total_count'] > 0 else 0) for k, v in metrics_sum.items() if k != 'total_count'}
    avg_metrics['loss'] = epoch_loss / iters

    return avg_metrics

def print_log(epoch: int, train_res: Dict, val_res: Dict, lr: float):
    if rank == 0:
        log.printlog(f'---------------- Epoch {epoch} ----------------')
        log.printlog(f"Loss (Train/Val): {train_res['loss']:.6f} / {val_res['loss']:.6f}")
        log.printlog(f"Pearson (Val):     {val_res['pearson_corr']:.4f}")
        log.printlog(f"SI-SDR (Val):      {val_res['si_sdr']:.2f} dB")
        log.printlog(f"ID Top-10 (Val):   {val_res['id_acc_top10']*100:.2f}%")

        log.train_writer.add_scalar('loss', train_res['loss'], epoch)
        log.val_writer.add_scalar('loss', val_res['loss'], epoch)
        log.val_writer.add_scalar('pearson_corr', val_res['pearson_corr'], epoch)
        log.val_writer.add_scalar('si_sdr', val_res['si_sdr'], epoch)
        log.val_writer.add_scalar('id_acc_top10', val_res['id_acc_top10'], epoch)
        log.train_writer.add_scalar('lr', lr, epoch)

def save_checkpoint(state, is_best: bool, filepath: str, filename: str):
    os.makedirs(filepath, exist_ok=True)
    if (state['epoch']) % 5 == 0 or state['epoch'] == 1:
        torch.save(state, os.path.join(filepath, filename))
        if rank == 0:
            log.printlog('Checkpoint saved!')
    if is_best and rank == 0:
        torch.save(state, os.path.join(filepath, 'model_best.pth'))
        log.printlog('Best model saved!')

def adjust_learning_rate_withWarmup(optimizer, epoch: int, args) -> float:
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    global rank, local_rank, log, device, scaler

    print(f'>>>>  Running on {device}  <<<<')

    model = Xmodel(embed_dim=3500, num_slots=args.num_slots, feature_dim=args.feature_dim, num_classes=args.num_classes)
    model.to(device)
    if rank == 0:
        log.printlog(model)

    # Updated to use the new RRUFFOnlineMixingDataset
    trainset = RRUFFOnlineMixingDataset(args.db_path, split='train', num_folds=args.num_folds, fold=args.fold,
                          encode_element=args.atom_embed, num_classes=args.num_classes)
    valset = RRUFFOnlineMixingDataset(args.db_path, split='val', num_folds=args.num_folds, fold=args.fold,
                        encode_element=args.atom_embed, num_classes=args.num_classes)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)

        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
        val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    # Early stopping state
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    patience = args.patience

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        train_res = run_one_epoch(model, train_loader, optimizer, epoch, mode='Train')
        val_res = run_one_epoch(model, val_loader, optimizer, epoch, mode='Eval')

        if rank == 0:
            print_log(epoch, train_res, val_res, optimizer.param_groups[0]['lr'])
            loss_val = val_res['loss']

            # Check validation loss
            is_best = False
            if loss_val < best_loss:
                best_loss = loss_val
                epochs_no_improve = 0
                is_best = True
            else:
                epochs_no_improve += 1

            save_checkpoint({'epoch': epoch,
                             'model': model.module.state_dict() if distributed else model.state_dict(),
                             'optimizer': optimizer.state_dict()}, is_best=is_best,
                            filepath=f'{log.get_path()}/checkpoints/',
                            filename=f'checkpoint_{epoch:04d}.pth')

            # Stop if patience is exhausted
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                early_stop = True
                break

        if early_stop:
            break

if __name__ == '__main__':
    rank, local_rank = 0, 0
    distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = torch.device("cuda", local_rank)
            dist.init_process_group(backend="nccl")
            print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
        else:
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")

        distributed = True
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # Enable verbose debugging
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--num_workers', default=16, type=int, metavar='N')
    parser.add_argument('--warmup_epochs', default=20, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--db_path', default='data/UniqCryLabeled.db', type=str,
                        help='Path to the metadata .db file')
    parser.add_argument('--npz_dir', default='data/UniqCry', type=str,
                        help='Directory containing the XRD .npz files')
    parser.add_argument('--atom_embed', type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--num_classes', default=100315, type=int, metavar='N')
    parser.add_argument('--num_slots', default=4, type=int, help='number of slots for phase separation')
    parser.add_argument('--feature_dim', default=256, type=int, help='dimension of slot features for retrieval')
    parser.add_argument('--patience', default=5, type=int, metavar='N', help='early stopping patience')
    parser.add_argument('--fold', default=0, type=int, help='Current fold index')
    parser.add_argument('--num_folds', default=5, type=int, help='Total number of folds')

    args = parser.parse_args()

    if rank == 0:
        log = Logger(val=True, append_str=f'_fold_{args.fold}')

    main()
    print('THE END')
