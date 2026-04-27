"""Evaluation runner for Transformer XRD separation baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from typing import Dict, Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'codex')}"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.transformer_family.models import build_transformer_family_baseline
from baselines.transformer_family.losses import calculate_pit_loss
from src.data.config import OnlineMixingConfig
from src.data.core import process_pattern
from src.data.online_mixing_dataset import create_online_mixing_dataloader
from src.utils.metrics import calculate_separation_metrics, calculate_sisdr


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def align_predictions(preds: torch.Tensor, best_perms: torch.Tensor) -> torch.Tensor:
    batch_size, num_sources, length = preds.shape
    inv_perms = torch.argsort(best_perms, dim=1)
    return torch.gather(preds, 1, inv_perms.unsqueeze(-1).expand(batch_size, num_sources, length))


def calculate_quant_mae(pred_patterns: torch.Tensor, target_patterns: torch.Tensor) -> float:
    pred_int = torch.clamp(pred_patterns.sum(dim=-1), min=0)
    target_int = torch.clamp(target_patterns.sum(dim=-1), min=0)
    pred_pct = pred_int / (pred_int.sum(dim=-1, keepdim=True) + 1e-8)
    target_pct = target_int / (target_int.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.abs(pred_pct - target_pct).mean().item() * 100.0


def pad_sources(
    targets: torch.Tensor,
    phase_ids: torch.Tensor | None,
    num_sources: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    current_sources = targets.shape[1]
    if current_sources == num_sources:
        return targets, phase_ids
    if current_sources > num_sources:
        targets = targets[:, :num_sources]
        if phase_ids is not None:
            phase_ids = phase_ids[:, :num_sources]
        return targets, phase_ids

    pad_n = num_sources - current_sources
    target_pad = torch.zeros(
        targets.shape[0],
        pad_n,
        targets.shape[2],
        dtype=targets.dtype,
        device=targets.device,
    )
    targets = torch.cat([targets, target_pad], dim=1)
    if phase_ids is not None:
        id_pad = torch.full(
            (phase_ids.shape[0], pad_n),
            -1,
            dtype=phase_ids.dtype,
            device=phase_ids.device,
        )
        phase_ids = torch.cat([phase_ids, id_pad], dim=1)
    return targets, phase_ids


def build_reference_bank(dataset, device: torch.device):
    print(f"Building reference bank for {len(dataset.indices)} crystals...", flush=True)
    ref_patterns = []
    ref_ids = []
    for crystal_id in tqdm(dataset.indices, desc="Building reference bank", file=sys.stdout, dynamic_ncols=True):
        patterns = dataset.get_crystal_patterns(int(crystal_id), max_samples=1)
        if not patterns:
            continue
        ref_patterns.append(process_pattern(patterns[0], dataset.xrd_length, norm_method="max"))
        ref_ids.append(int(crystal_id))

    if not ref_patterns:
        print("Reference bank is empty; identification metrics will be disabled.", flush=True)
        return None, None
    print(f"Reference bank ready: {len(ref_patterns)} patterns.", flush=True)
    return torch.stack(ref_patterns).to(device), torch.tensor(ref_ids, dtype=torch.long, device=device)


def calculate_identification_topk(
    aligned_preds: torch.Tensor,
    phase_ids: torch.Tensor,
    target_is_active: torch.Tensor,
    ref_bank: torch.Tensor,
    ref_ids: torch.Tensor,
    topk: Iterable[int] = tuple(range(1, 11)),
) -> Dict[str, float]:
    topk = tuple(topk)
    batch_size, num_sources, length = aligned_preds.shape
    valid = (phase_ids != -1) & (target_is_active > 0.5)
    if valid.sum().item() == 0:
        return {f"id_acc_top{k}": 0.0 for k in topk}

    preds_flat = aligned_preds.reshape(batch_size * num_sources, length)
    ids_flat = phase_ids.reshape(batch_size * num_sources)
    valid_flat = valid.reshape(batch_size * num_sources)

    pred_norm = torch.nn.functional.normalize(preds_flat[valid_flat], p=2, dim=1)
    ref_norm = torch.nn.functional.normalize(ref_bank, p=2, dim=1)
    sim = torch.matmul(pred_norm, ref_norm.T)
    max_k = min(max(topk), ref_bank.shape[0])
    _, indices = torch.topk(sim, k=max_k, dim=1)
    retrieved = ref_ids[indices]
    target_ids = ids_flat[valid_flat].unsqueeze(1)

    metrics = {}
    for k in topk:
        k_eff = min(k, max_k)
        hit = (retrieved[:, :k_eff] == target_ids).any(dim=1).float().mean().item()
        metrics[f"id_acc_top{k}"] = hit
    return metrics


def evaluate_model(model, test_loader, device, args, config: Dict) -> Dict[str, float]:
    model.eval()
    ref_bank = ref_ids = None
    if not args.disable_identification:
        ref_bank, ref_ids = build_reference_bank(test_loader.dataset, device)
    print(f"Testing {len(test_loader)} batches on {device}...", flush=True)

    accumulated = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "rwp": 0.0,
        "pearson_corr": 0.0,
        "sir": 0.0,
        "sar": 0.0,
        "delta_2theta": 0.0,
        "fwhm_error": 0.0,
        "intensity_consistency": 0.0,
        "act_acc": 0.0,
        "act_f1": 0.0,
        "act_precision": 0.0,
        "act_recall": 0.0,
        "act_exact_match": 0.0,
        "quant_mae": 0.0,
        **{f"id_acc_top{k}": 0.0 for k in range(1, 11)},
    }
    steps = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", file=sys.stdout, dynamic_ncols=True)):
            if args.quick and batch_idx >= 5:
                break

            mix = batch["multiphase_xrd"].to(device, non_blocking=True)
            targets = batch["single_xrds"].to(device, non_blocking=True)
            phase_ids = batch["phase_ids"].to(device, non_blocking=True)

            with autocast_context(device):
                preds, activity_logits = model(mix)
                targets, phase_ids = pad_sources(targets, phase_ids, preds.shape[1])
                sep_loss, best_perms = calculate_pit_loss(preds, targets)

            aligned_preds = align_predictions(preds, best_perms)
            inv_perms = torch.argsort(best_perms, dim=1)
            metrics = calculate_separation_metrics(
                aligned_preds,
                targets,
                two_theta_range=(5.0, 90.0),
                calc_detailed=True,
            )

            target_energy = (targets**2).sum(dim=-1)
            target_is_active = (target_energy > 1e-6).float()
            aligned_act_logits = torch.gather(activity_logits, 1, inv_perms)
            act_pred = (torch.sigmoid(aligned_act_logits) > args.activity_threshold).float()

            tp = (act_pred * target_is_active).sum()
            fp = (act_pred * (1 - target_is_active)).sum()
            fn = ((1 - act_pred) * target_is_active).sum()
            precision = (tp / (tp + fp + 1e-8)).item()
            recall = (tp / (tp + fn + 1e-8)).item()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            acc = (act_pred == target_is_active).float().mean().item()
            exact = (act_pred == target_is_active).all(dim=1).float().mean().item()

            id_metrics = {f"id_acc_top{k}": 0.0 for k in range(1, 11)}
            if ref_bank is not None and ref_ids is not None:
                id_metrics = calculate_identification_topk(
                    aligned_preds,
                    phase_ids,
                    target_is_active,
                    ref_bank,
                    ref_ids,
                )

            accumulated["loss"] += sep_loss.item()
            accumulated["si_sdr"] += calculate_sisdr(aligned_preds, targets)
            accumulated["rwp"] += metrics["rwp"]
            accumulated["pearson_corr"] += metrics["pearson_corr"]
            accumulated["sir"] += metrics["sir"]
            accumulated["sar"] += metrics["sar"]
            accumulated["delta_2theta"] += metrics["delta_2theta"]
            accumulated["fwhm_error"] += metrics["fwhm_error"]
            accumulated["intensity_consistency"] += metrics["intensity_ratio_consistency"]
            accumulated["act_acc"] += acc
            accumulated["act_f1"] += f1
            accumulated["act_precision"] += precision
            accumulated["act_recall"] += recall
            accumulated["act_exact_match"] += exact
            accumulated["quant_mae"] += calculate_quant_mae(aligned_preds, targets)
            for key, value in id_metrics.items():
                accumulated[key] += value
            steps += 1

    result = {k: v / max(1, steps) for k, v in accumulated.items()}
    result["baseline_name"] = config.get("baseline_name", args.baseline_name or "unknown")
    result["checkpoint"] = args.checkpoint
    result["split"] = args.split
    result["min_k"] = args.min_k
    result["max_k"] = args.max_k
    return result


def visualize(model, dataset, device, args, ref_bank=None, ref_ids=None) -> None:
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    num_samples = min(args.num_vis, len(dataset))
    if num_samples <= 0:
        return
    print(f"Writing {num_samples} visualizations to {args.save_dir}...", flush=True)
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for out_idx, sample_idx in enumerate(tqdm(indices, desc="Visualizing", file=sys.stdout, dynamic_ncols=True)):
        sample = dataset[int(sample_idx)]
        mix = sample["multiphase_xrd"].unsqueeze(0).to(device)
        targets = sample["single_xrds"].unsqueeze(0).to(device)
        phase_ids = sample["phase_ids"].numpy()

        with torch.no_grad(), autocast_context(device):
            preds, activity_logits = model(mix)
            targets, _ = pad_sources(targets, None, preds.shape[1])
            _, best_perms = calculate_pit_loss(preds, targets)

        pred_ordered = align_predictions(preds, best_perms)[0].float()
        act_ordered = torch.sigmoid(torch.gather(activity_logits, 1, torch.argsort(best_perms, dim=1)))[0].float()

        pred_np = pred_ordered.cpu().numpy()
        target_np = targets[0].float().cpu().numpy()
        mix_np = mix[0].float().cpu().numpy()
        act_np = act_ordered.cpu().numpy()
        if len(phase_ids) < args.num_phases:
            phase_ids = np.pad(phase_ids, (0, args.num_phases - len(phase_ids)), constant_values=-1)

        fig, axes = plt.subplots(args.num_phases + 1, 1, figsize=(14, 3 * (args.num_phases + 1)), sharex=True)
        axes = np.asarray(axes)
        axes[0].plot(mix_np, color="black", linewidth=1.4, alpha=0.75, label="Mixture")
        axes[0].plot(pred_np.sum(axis=0), color="red", linewidth=1.1, alpha=0.8, label="Pred sum")
        axes[0].set_title(f"Sample {sample_idx}")
        axes[0].legend(fontsize="small")

        cmap = plt.get_cmap("tab10")
        for k in range(args.num_phases):
            ax = axes[k + 1]
            color = cmap(k % 10)
            pid = int(phase_ids[k])
            if target_np[k].max() > 1e-4:
                ax.fill_between(range(len(target_np[k])), target_np[k], color=color, alpha=0.2, label=f"GT {k} id={pid}")
                ax.plot(target_np[k], color=color, alpha=0.45, linewidth=1.0)
            else:
                ax.text(0.02, 0.65, f"GT {k}: silent", transform=ax.transAxes, color=color)
            ax.plot(pred_np[k], color=color, linewidth=1.2, label=f"Pred {k} p={act_np[k]:.2f}")
            ax.legend(fontsize="x-small")

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"test_sample_{out_idx}_{sample_idx}.png"), dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer XRD separation baselines")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_name", choices=["transformer", "itransformer", "patchtst"], default=None)
    parser.add_argument("--mae_checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=os.environ.get("PATH_DATA_SINGLEPHASE"))
    parser.add_argument("--crystal_db", type=str, default=os.environ.get("PATH_DATA_CRYSTAL_DB", ""))
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("PATH_OUTPUT_BASELINE_EVAL", "test_results/transformer"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num_vis", type=int, default=20)
    parser.add_argument("--num_phases", type=int, default=None)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument("--k_weights", type=float, nargs="+", default=None)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--disable_identification", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    baseline_name = args.baseline_name or config.get("baseline_name")
    if baseline_name is None:
        raise ValueError("--baseline_name is required when checkpoint config does not contain it")

    mae_checkpoint = (
        args.mae_checkpoint
        or os.environ.get("PATH_CKPT_PRETRAIN")
        or config.get("mae_checkpoint")
    )
    if not mae_checkpoint:
        raise ValueError("MAE checkpoint is required via --mae_checkpoint, PATH_CKPT_PRETRAIN, or checkpoint config")

    num_phases = args.num_phases or int(config.get("num_phases", 4))
    args.num_phases = num_phases
    xrd_length = int(config.get("xrd_length", 3500))

    print(
        f"Building model: baseline={baseline_name}, num_phases={num_phases}, xrd_length={xrd_length}",
        flush=True,
    )
    model = build_transformer_family_baseline(
        name=baseline_name,
        mae_checkpoint=mae_checkpoint,
        num_sources=num_phases,
        xrd_length=xrd_length,
        patch_len=int(config.get("patch_len", 50)),
        stride=int(config.get("stride", 25)),
        d_model=config.get("d_model", None),
        n_heads=config.get("n_heads", None),
        n_layers=config.get("n_layers", None),
        dropout=float(config.get("dropout", 0.1)),
        freeze_backbone=bool(config.get("freeze_backbone", False)),
        output_activation=config.get("output_activation", "relu"),
        transformer_decoder_layers=int(config.get("transformer_decoder_layers", 4)),
        transformer_d_ff=config.get("transformer_d_ff", None),
        itransformer_dim=int(config.get("itransformer_dim", 256)),
        itransformer_layers=int(config.get("itransformer_layers", 2)),
        itransformer_heads=int(config.get("itransformer_heads", 8)),
        itransformer_d_ff=config.get("itransformer_d_ff", None),
    )
    model.load_state_dict(strip_module_prefix(ckpt["model_state_dict"]))

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    print(f"Model loaded on device: {device}", flush=True)

    max_k = args.max_k if args.max_k is not None else num_phases
    args.max_k = max_k
    data_kwargs = {
        "XRD_LENGTH": xrd_length,
        "MIN_K": args.min_k,
        "MAX_K": max_k,
        "AUGMENT": False,
        "SEED": int(config.get("seed", 42)),
    }
    if args.k_weights is not None:
        data_kwargs["K_WEIGHTS"] = tuple(args.k_weights)
        data_kwargs["K_DISTRIBUTION"] = "weighted"
    data_config = OnlineMixingConfig(**data_kwargs)

    data_dir = args.data_dir or config.get("singlephase_xrd_db")
    if not data_dir:
        raise ValueError("Data dir is required via --data_dir or checkpoint config")

    print(
        f"Building dataloader: split={args.split}, min_k={args.min_k}, max_k={max_k}, batch_size={args.batch_size}",
        flush=True,
    )
    test_loader = create_online_mixing_dataloader(
        data_dir,
        args.crystal_db or config.get("crystal_db", ""),
        data_config,
        split=args.split,
        train_ratio=0.0,
        val_ratio=0.0,
        batch_size=args.batch_size,
        distributed=False,
    )
    print(f"Dataloader ready: {len(test_loader.dataset)} samples, {len(test_loader)} batches.", flush=True)

    log_path = os.path.join(args.save_dir, "evaluation.log")
    with open(log_path, "w") as log_file:
        log_file.write(f"checkpoint: {args.checkpoint}\n")
        log_file.write(f"baseline_name: {baseline_name}\n")
        log_file.write(f"data_dir: {data_dir}\n")
        log_file.write(f"min_k: {args.min_k}, max_k: {max_k}\n")

    metrics = evaluate_model(model, test_loader, device, args, {**config, "baseline_name": baseline_name})
    metrics_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    visualize(model, test_loader.dataset, device, args)
    print(f"Visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    main()
