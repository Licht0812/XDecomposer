import argparse
import math
import os
from typing import Dict
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import ASEDataset
from model.XQueryer import Xmodel
from util.logger import Logger

import argparse
import math
import os
from typing import Dict
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from model.dataset import ASEDataset
from model.XQueryer import Xmodel
from util.logger import Logger

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

def run_one_epoch(model, dataloader, optimizer, epoch, mode):
    if mode == 'Train':
        model.train()
        desc = 'Training... '
    else:
        model.eval()
        desc = 'Evaluating... '

    epoch_loss = 0
    metrics_sum = {
        'rwp': 0, 'sisdr': 0, 'ratio_mae': 0, 'top10_acc': 0, 'total_count': 0
    }
    
    if args.progress_bar:
        pbar = tqdm(total=len(dataloader.dataset), desc=desc, unit='data')
    iters = len(dataloader)

    criterion_mse = torch.nn.MSELoss()
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
                    valid_gt_mask = gt_ratios[b] > 1e-6
                    num_valid_gt = valid_gt_mask.sum().item()
                    
                    if num_valid_gt > 0:
                        # Cost matrix: MSE between all pairs of (pred_xrd, gt_xrd)
                        # Convert to float32 as cdist might not support float16
                        cost_matrix = torch.cdist(pred_xrds[b].float(), gt_xrds[b][valid_gt_mask].float(), p=2).cpu().detach().numpy()
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
                        for r, c in zip(row_ind, col_ind):
                            gt_idx = torch.where(valid_gt_mask)[0][c]
                            total_loss += criterion_mse(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            total_loss += criterion_mse(pred_ratios[b, r], gt_ratios[b, gt_idx])
                            total_loss += criterion_cls(feat_logits[b, r].unsqueeze(0), gt_ids[b, gt_idx].unsqueeze(0))
                    else:
                        total_loss += pred_ratios[b].pow(2).sum()

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
                    valid_gt_mask = gt_ratios[b] > 1e-6
                    num_valid_gt = valid_gt_mask.sum().item()
                    if num_valid_gt > 0:
                        # Convert to float32 for cdist
                        cost_matrix = torch.cdist(pred_xrds[b].float(), gt_xrds[b][valid_gt_mask].float(), p=2).cpu().numpy()
                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
                        matched_logits = []
                        matched_targets = []
                        for r, c in zip(row_ind, col_ind):
                            gt_idx = torch.where(valid_gt_mask)[0][c]
                            total_loss += criterion_mse(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            total_loss += criterion_mse(pred_ratios[b, r], gt_ratios[b, gt_idx])
                            total_loss += criterion_cls(feat_logits[b, r].unsqueeze(0), gt_ids[b, gt_idx].unsqueeze(0))
                            
                            metrics_sum['rwp'] += calculate_rwp(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['sisdr'] += calculate_sisdr(pred_xrds[b, r], gt_xrds[b, gt_idx])
                            metrics_sum['ratio_mae'] += torch.abs(pred_ratios[b, r] - gt_ratios[b, gt_idx]).item()
                            
                            matched_logits.append(feat_logits[b, r])
                            matched_targets.append(gt_ids[b, gt_idx])
                            metrics_sum['total_count'] += 1
                        
                        if matched_logits:
                            metrics_sum['top10_acc'] += get_id_acc_topk(torch.stack(matched_logits), torch.stack(matched_targets), k=10) * len(matched_logits)
                    else:
                        total_loss += pred_ratios[b].pow(2).sum()
                
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
        log.printlog(f"RWP (Val):         {val_res['rwp']:.4f}")
        log.printlog(f"SI-SDR (Val):      {val_res['sisdr']:.2f} dB")
        log.printlog(f"Ratio MAE (Val):   {val_res['ratio_mae']*100:.2f}%")
        log.printlog(f"Top-10 Acc (Val):  {val_res['top10_acc']*100:.2f}%")

        log.train_writer.add_scalar('loss', train_res['loss'], epoch)
        log.val_writer.add_scalar('loss', val_res['loss'], epoch)
        log.val_writer.add_scalar('rwp', val_res['rwp'], epoch)
        log.val_writer.add_scalar('sisdr', val_res['sisdr'], epoch)
        log.val_writer.add_scalar('ratio_mae', val_res['ratio_mae'], epoch)
        log.val_writer.add_scalar('top10_acc', val_res['top10_acc'], epoch)
        log.train_writer.add_scalar('lr', lr, epoch)

def save_checkpoint(state, is_best: bool, filepath: str, filename: str):
    if (state['epoch']) % 5 == 0 or state['epoch'] == 1:
        os.makedirs(filepath, exist_ok=True)
        torch.save(state, os.path.join(filepath, filename))
        if rank == 0:
            log.printlog('Checkpoint saved!')
            if is_best:
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

    # Updated to use the new ASEDataset with ID-based splitting
    trainset = ASEDataset(args.db_path, args.npz_dir, mode='train', 
                          encode_element=args.atom_embed, num_classes=args.num_classes)
    valset = ASEDataset(args.db_path, args.npz_dir, mode='val', 
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

    # 早停止相关变量
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
            save_checkpoint({'epoch': epoch,
                             'model': model.module.state_dict() if distributed else model.state_dict(),
                             'optimizer': optimizer.state_dict()}, is_best=False,
                            filepath=f'{log.get_path()}/checkpoints/',
                            filename=f'checkpoint_{epoch:04d}.pth')

            # 检查验证集损失是否有改善
            if loss_val < best_loss:
                best_loss = loss_val
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # 检查是否需要早停止
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

    # 设置环境变量以获取详细调试信息
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N')
    parser.add_argument('--num_workers', default=16, type=int, metavar='N')
    parser.add_argument('--warmup_epochs', default=20, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--db_path', default='/data/group/project1/Crystal/UniqCryLabeled.db', type=str,
                        help='Path to the metadata .db file')
    parser.add_argument('--npz_dir', default='/data/group/project1/Crystal/UniqCry', type=str,
                        help='Directory containing the XRD .npz files')
    parser.add_argument('--atom_embed', type=lambda x: (str(x).lower() in ['true','1']), default=True)
    parser.add_argument('--num_classes', default=100315, type=int, metavar='N')
    parser.add_argument('--num_slots', default=4, type=int, help='number of slots for phase separation')
    parser.add_argument('--feature_dim', default=256, type=int, help='dimension of slot features for retrieval')
    parser.add_argument('--patience', default=5, type=int, metavar='N', help='early stopping patience')

    args = parser.parse_args()

    if rank == 0:
        log = Logger(val=True)

    main()
    print('THE END')
