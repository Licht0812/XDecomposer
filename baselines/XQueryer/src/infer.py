import argparse
import gc
import json
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.XQueryer import Xmodel
from model.dataset import ASEDataset, OnlineMixingConfig


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def calculate_pearson_correlation(pred_pattern: torch.Tensor, target_pattern: torch.Tensor) -> float:
    """Calculate Pearson correlation."""
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
    """Calculate SI-SDR in dB."""
    if pred.ndim > 2:
        length = pred.shape[-1]
        pred = pred.reshape(-1, length)
        target = target.reshape(-1, length)
    elif pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    dot_product = torch.sum(pred * target, dim=-1)
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    alpha = dot_product / target_energy
    e_target = alpha.unsqueeze(-1) * target
    e_res = pred - e_target
    signal_energy = torch.sum(e_target ** 2, dim=-1)
    noise_energy = torch.sum(e_res ** 2, dim=-1) + eps
    ratio = torch.clamp(signal_energy / noise_energy, min=1e-10)
    return (10 * torch.log10(ratio)).mean().item()


def calculate_sir_sar(
    pred: torch.Tensor,
    target: torch.Tensor,
    interference: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """Calculate SIR and SAR."""
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    artifacts = pred - target
    target_energy = torch.sum(target ** 2, dim=-1) + eps
    artifacts_energy = torch.sum(artifacts ** 2, dim=-1) + eps
    sar = 10 * torch.log10(target_energy / artifacts_energy)

    if interference is None:
        interference = artifacts
    elif interference.ndim == 1:
        interference = interference.unsqueeze(0)

    interference_energy = torch.sum(interference ** 2, dim=-1) + eps
    sir = 10 * torch.log10(target_energy / interference_energy)
    return sir.mean().item(), sar.mean().item()


def weighted_xrd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 10.0,
    lambda_cos: float = 0.2,
    lambda_geo: float = 0.1,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    focal = 1.0 + alpha * target
    loss_l1 = (torch.abs(pred - target) * focal).mean()
    loss_cos = 1.0 - F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0), dim=-1, eps=eps).mean()

    pred_sqrt = torch.sqrt(torch.clamp(pred, min=eps))
    target_sqrt = torch.sqrt(torch.clamp(target, min=eps))
    grad1 = torch.abs((pred_sqrt[1:] - pred_sqrt[:-1]) - (target_sqrt[1:] - target_sqrt[:-1])).mean()
    grad2 = torch.abs(
        (pred_sqrt[2:] - 2 * pred_sqrt[1:-1] + pred_sqrt[:-2])
        - (target_sqrt[2:] - 2 * target_sqrt[1:-1] + target_sqrt[:-2])
    ).mean()
    loss_geo = grad1 + beta * grad2
    return loss_l1 + lambda_cos * loss_cos + lambda_geo * loss_geo


def build_matching_cost(pred_xrds: torch.Tensor, gt_xrds: torch.Tensor) -> torch.Tensor:
    cost_matrix = torch.zeros(pred_xrds.size(0), gt_xrds.size(0), device=pred_xrds.device)
    for pred_idx in range(pred_xrds.size(0)):
        for target_idx in range(gt_xrds.size(0)):
            cost_matrix[pred_idx, target_idx] = weighted_xrd_loss(pred_xrds[pred_idx], gt_xrds[target_idx]).detach()
    return cost_matrix


def compute_sample_objective(
    pred_xrds: torch.Tensor,
    pred_ratios: torch.Tensor,
    feat_logits: torch.Tensor,
    gt_xrds: torch.Tensor,
    gt_ratios: torch.Tensor,
    gt_ids: torch.Tensor,
    mixture: torch.Tensor,
    criterion_ratio: torch.nn.Module,
    criterion_cls: torch.nn.Module,
    lambda_ratio: float = 1.0,
    lambda_cls: float = 0.1,
    lambda_mix: float = 1.0,
    lambda_empty: float = 0.5,
):
    valid_gt_mask = gt_ratios > 1e-6
    valid_gt_indices = torch.where(valid_gt_mask)[0]
    matched_rows = torch.zeros(pred_xrds.size(0), dtype=torch.bool, device=pred_xrds.device)
    sample_loss = lambda_mix * F.l1_loss(pred_xrds.sum(dim=0), mixture)
    row_ind = np.array([], dtype=np.int64)
    col_ind = np.array([], dtype=np.int64)

    if valid_gt_indices.numel() > 0:
        gt_active_xrds = gt_xrds[valid_gt_indices]
        cost_matrix = build_matching_cost(pred_xrds.float(), gt_active_xrds.float()).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for pred_idx, target_rel_idx in zip(row_ind, col_ind):
            gt_idx = valid_gt_indices[target_rel_idx]
            sample_loss = sample_loss + weighted_xrd_loss(pred_xrds[pred_idx], gt_xrds[gt_idx])
            sample_loss = sample_loss + lambda_ratio * criterion_ratio(pred_ratios[pred_idx], gt_ratios[gt_idx])
            sample_loss = sample_loss + lambda_cls * criterion_cls(
                feat_logits[pred_idx].unsqueeze(0),
                gt_ids[gt_idx].unsqueeze(0),
            )
            matched_rows[pred_idx] = True

    if (~matched_rows).any():
        sample_loss = sample_loss + lambda_empty * pred_xrds[~matched_rows].abs().mean()
        sample_loss = sample_loss + lambda_empty * pred_ratios[~matched_rows].mean()

    return sample_loss, row_ind, col_ind, valid_gt_indices


def find_xrd_peaks_batch(
    xrd_patterns: torch.Tensor,
    height_threshold: float = 0.05,
    distance: int = 10,
) -> torch.Tensor:
    """Detect peaks in batched XRD patterns."""
    if xrd_patterns.dim() == 1:
        xrd_patterns = xrd_patterns.unsqueeze(0)

    xrd_patterns = xrd_patterns.float()
    mask = xrd_patterns > height_threshold
    x_pad = F.pad(xrd_patterns, (1, 1), value=-1e9)
    is_local_max = (xrd_patterns > x_pad[:, :-2]) & (xrd_patterns > x_pad[:, 2:])
    mask = mask & is_local_max

    if distance > 0:
        kernel_size = 2 * distance + 1
        x_pad_pool = F.pad(xrd_patterns.unsqueeze(1), (distance, distance), value=-1e9)
        max_in_window = F.max_pool1d(x_pad_pool, kernel_size=kernel_size, stride=1).squeeze(1)
        mask = mask & (torch.abs(xrd_patterns - max_in_window) < 1e-6)

    return mask


def _get_batch_peak_indices(mask: torch.Tensor) -> torch.Tensor:
    batch_size, length = mask.shape
    device = mask.device
    idx_map = torch.arange(length, device=device).unsqueeze(0).expand(batch_size, length)
    masked_idxs = torch.where(mask, idx_map, torch.tensor(length, device=device))
    sorted_idxs, _ = torch.sort(masked_idxs, dim=1)
    max_peaks = mask.sum(dim=1).max().item()
    if max_peaks == 0:
        return torch.full((batch_size, 0), -1, device=device, dtype=torch.long)

    dense_idxs = sorted_idxs[:, :max_peaks].clone()
    dense_idxs[dense_idxs == length] = -1
    return dense_idxs


def calculate_peak_shift_delta_2theta(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    two_theta_range: Tuple[float, float] = (10.0, 80.0),
    tolerance: int = 10,
    **peak_args: Any,
) -> float:
    """Calculate mean peak shift in delta 2-theta."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    _, length = pred_pattern.shape
    pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_args)
    target_mask = find_xrd_peaks_batch(target_pattern, **peak_args)
    pred_indices = _get_batch_peak_indices(pred_mask)
    target_indices = _get_batch_peak_indices(target_mask)
    if pred_indices.shape[1] == 0 or target_indices.shape[1] == 0:
        return 0.0

    min_2theta, max_2theta = two_theta_range
    pred_2theta = min_2theta + (max_2theta - min_2theta) * pred_indices.float() / (length - 1)
    target_2theta = min_2theta + (max_2theta - min_2theta) * target_indices.float() / (length - 1)
    dists = torch.abs(pred_2theta.unsqueeze(2) - target_2theta.unsqueeze(1))
    valid_pairs = (pred_indices != -1).unsqueeze(2) & (target_indices != -1).unsqueeze(1)
    tol_val = (max_2theta - min_2theta) * tolerance / (length - 1)
    dists = dists.masked_fill(~(valid_pairs & (dists <= tol_val)), float("inf"))
    min_dists, _ = dists.min(dim=2)
    matched = min_dists != float("inf")
    if not matched.any():
        return 0.0

    return (min_dists[matched].sum() / matched.sum()).item()


def calculate_fwhm_error(
    pred_pattern: torch.Tensor,
    target_pattern: torch.Tensor,
    window_size: int = 50,
    **peak_args: Any,
) -> float:
    """Calculate FWHM error."""
    if pred_pattern.dim() == 1:
        pred_pattern = pred_pattern.unsqueeze(0)
        target_pattern = target_pattern.unsqueeze(0)

    _, length = pred_pattern.shape
    device = pred_pattern.device
    pred_mask = find_xrd_peaks_batch(pred_pattern, **peak_args)
    target_mask = find_xrd_peaks_batch(target_pattern, **peak_args)
    pred_indices = _get_batch_peak_indices(pred_mask)
    target_indices = _get_batch_peak_indices(target_mask)
    if pred_indices.shape[1] == 0 or target_indices.shape[1] == 0:
        return 0.0

    dists = torch.abs(pred_indices.unsqueeze(2).float() - target_indices.unsqueeze(1).float())
    valid = (pred_indices != -1).unsqueeze(2) & (target_indices != -1).unsqueeze(1) & (dists <= 10)
    min_dists, min_target_rel = dists.masked_fill(~valid, float("inf")).min(dim=2)
    matched = min_dists != float("inf")
    if not matched.any():
        return 0.0

    batch_idx, pred_rel_idx = torch.nonzero(matched, as_tuple=True)
    pred_peaks = pred_indices[batch_idx, pred_rel_idx]
    target_peaks = target_indices[batch_idx, min_target_rel[batch_idx, pred_rel_idx]]

    def compute_fwhm(patterns: torch.Tensor, peak_idx: torch.Tensor) -> torch.Tensor:
        count = peak_idx.shape[0]
        half_window = window_size // 2
        offsets = torch.arange(-half_window, half_window + 1, device=device)
        window_indices = torch.clamp(peak_idx.unsqueeze(1) + offsets.unsqueeze(0), 0, length - 1)
        windows = torch.gather(patterns, 1, window_indices)
        half_max = patterns[torch.arange(count, device=device), peak_idx].unsqueeze(1) / 2.0

        left = torch.flip(windows[:, :half_window], [1]) <= half_max
        left_dist = torch.where(
            left.any(dim=1),
            torch.argmax(left.float(), dim=1) + 1,
            torch.tensor(half_window, device=device),
        )

        right = windows[:, half_window + 1:] <= half_max
        right_dist = torch.where(
            right.any(dim=1),
            torch.argmax(right.float(), dim=1) + 1,
            torch.tensor(right.shape[1], device=device),
        )
        return (left_dist + right_dist).float()

    pred_fwhm = compute_fwhm(pred_pattern[batch_idx], pred_peaks)
    target_fwhm = compute_fwhm(target_pattern[batch_idx], target_peaks)
    return torch.abs(pred_fwhm - target_fwhm).mean().item()


def init_metric_sums() -> Dict[str, float]:
    """Initialize metric accumulators."""
    return {
        "loss_sum": 0.0,
        "pearson_corr_sum": 0.0,
        "si_sdr_sum": 0.0,
        "sir_sum": 0.0,
        "sar_sum": 0.0,
        "delta_2theta_sum": 0.0,
        "fwhm_error_sum": 0.0,
        "total_samples": 0,
        "total_phases": 0,
        **{f"id_acc_top{k}_hits": 0.0 for k in range(1, 11)},
    }


def finalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Convert accumulators to a flat summary."""
    sample_count = max(1, int(metrics["total_samples"]))
    phase_count = max(1, int(metrics["total_phases"]))
    summary = {
        "loss": metrics["loss_sum"] / sample_count,
        "si_sdr": metrics["si_sdr_sum"] / phase_count,
        "pearson_corr": metrics["pearson_corr_sum"] / phase_count,
        "sir": metrics["sir_sum"] / phase_count,
        "sar": metrics["sar_sum"] / phase_count,
        "delta_2theta": metrics["delta_2theta_sum"] / phase_count,
        "fwhm_error": metrics["fwhm_error_sum"] / phase_count,
    }
    for k in range(1, 11):
        summary[f"id_acc_top{k}"] = metrics[f"id_acc_top{k}_hits"] / phase_count
    return summary


def run_one_epoch(model, dataloader, device, save_path: str = "inference_results.json", limit: int = 0):
    """Run evaluation and save metric summaries."""
    model.eval()
    metrics = init_metric_sums()
    criterion_ratio = torch.nn.L1Loss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    total_to_eval = len(dataloader.dataset) if limit == 0 else min(limit, len(dataloader.dataset))
    pbar = tqdm(total=total_to_eval, desc="Evaluating", unit="sample")

    with torch.no_grad():
        for batch in dataloader:
            if limit > 0 and metrics["total_samples"] >= limit:
                break

            intensity = batch["intensity"].to(device)
            element = batch["element"].to(device)
            gt_xrds = batch["gt_xrds"].to(device)
            gt_ratios = batch["gt_ratios"].to(device)
            gt_ids = batch["gt_ids"].to(device)

            with autocast_context(device):
                outputs = model(intensity, element)
                pred_xrds = outputs["xrds"]
                pred_ratios = outputs["ratios"]
                feat_logits = outputs["feat_logits"]

            batch_size = intensity.size(0)
            for batch_idx in range(batch_size):
                if limit > 0 and metrics["total_samples"] >= limit:
                    break

                valid_gt_mask = gt_ratios[batch_idx] > 1e-6
                valid_gt_indices = torch.where(valid_gt_mask)[0]
                if valid_gt_indices.numel() == 0:
                    continue

                sample_loss, row_ind, col_ind, valid_gt_indices = compute_sample_objective(
                    pred_xrds[batch_idx],
                    pred_ratios[batch_idx],
                    feat_logits[batch_idx],
                    gt_xrds[batch_idx],
                    gt_ratios[batch_idx],
                    gt_ids[batch_idx],
                    intensity[batch_idx],
                    criterion_ratio,
                    criterion_cls,
                )
                if len(row_ind) == 0:
                    continue

                metrics["loss_sum"] += sample_loss.item()
                metrics["total_samples"] += 1

                for pred_slot, rel_target_idx in zip(row_ind, col_ind):
                    gt_idx = valid_gt_indices[rel_target_idx]
                    target_id = int(gt_ids[batch_idx, gt_idx].item())
                    logits = feat_logits[batch_idx, pred_slot]
                    max_k = min(10, logits.shape[-1])
                    top_indices = torch.topk(logits, k=max_k).indices.cpu().tolist()

                    metrics["total_phases"] += 1
                    for k in range(1, 11):
                        if target_id in top_indices[: min(k, max_k)]:
                            metrics[f"id_acc_top{k}_hits"] += 1.0

                    pred_pattern = pred_xrds[batch_idx, pred_slot]
                    target_pattern = gt_xrds[batch_idx, gt_idx]
                    other_mask = valid_gt_mask.clone()
                    other_mask[gt_idx] = False
                    interference = (
                        gt_xrds[batch_idx, other_mask].sum(dim=0)
                        if other_mask.any()
                        else torch.zeros_like(target_pattern)
                    )
                    sir, sar = calculate_sir_sar(pred_pattern, target_pattern, interference=interference)

                    metrics["pearson_corr_sum"] += calculate_pearson_correlation(pred_pattern, target_pattern)
                    metrics["si_sdr_sum"] += calculate_sisdr(pred_pattern, target_pattern)
                    metrics["sir_sum"] += sir
                    metrics["sar_sum"] += sar
                    metrics["delta_2theta_sum"] += calculate_peak_shift_delta_2theta(pred_pattern, target_pattern)
                    metrics["fwhm_error_sum"] += calculate_fwhm_error(pred_pattern, target_pattern)

                pbar.update(1)

                if metrics["total_samples"] % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    pbar.close()

    summary = finalize_metrics(metrics)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n--- Evaluation Summary ---")
    for key, value in summary.items():
        print(f"{key:20s}: {value:.4f}")
    print("Results saved.")


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Xmodel(
        embed_dim=3500,
        num_slots=args.num_slots,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
    )
    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.load_path}")

    config = OnlineMixingConfig()
    if args.num_phases in [2, 3, 4]:
        print(f"Using fixed {args.num_phases}-phase mixtures.")
        config.MIN_K = args.num_phases
        config.MAX_K = args.num_phases
    else:
        print("Using mixed 2-4 phase mixtures.")

    testset = ASEDataset(
        args.db_path,
        args.npz_dir,
        mode="test",
        encode_element=args.atom_embed,
        num_classes=args.num_classes,
        config=config,
    )

    batch_size = args.batch_size if torch.cuda.is_available() else 1
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    output_dir = os.path.dirname(args.load_path)
    if "checkpoints" in output_dir:
        output_dir = os.path.dirname(output_dir)
    if args.num_phases > 0:
        save_path = os.path.join(output_dir, f"inference_results_{args.num_phases}_phases.json")
    else:
        save_path = os.path.join(output_dir, "inference_results.json")
    print("Preparing results output.")

    run_one_epoch(model, test_loader, device, save_path=save_path, limit=args.limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, choices=["cuda:0", "cpu"])
    parser.add_argument("--db_path", default="data/UniqCryLabeled.db", type=str)
    parser.add_argument("--npz_dir", default="data/UniqCry", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--atom_embed", default=True, type=bool)
    parser.add_argument("--load_path", default="checkpoints/xqueryer/latest.pth", type=str)
    parser.add_argument("--num_classes", default=100315, type=int)
    parser.add_argument("--num_slots", default=4, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--entries_dict", default="./entries_dict.json", type=str)
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--num_phases", type=int, default=0)

    args = parser.parse_args()
    main()
