"""RRUFF k-fold evaluation for Transformer-family baselines."""

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

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.transformer_family.evaluate import (  # noqa: E402
    align_predictions,
    calculate_identification_topk,
    pad_sources,
    strip_module_prefix,
)
from baselines.transformer_family.losses import calculate_pit_loss  # noqa: E402
from baselines.transformer_family.models import build_transformer_family_baseline  # noqa: E402
from src.data.rruff_dataset import baseline_als, create_rruff_dataloader  # noqa: E402
from src.utils.metrics import calculate_separation_metrics, calculate_sisdr  # noqa: E402

def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()

def ensure_rruff_cache(
    rruff_db_path: str,
    target_length: int,
    theta_min: float = 10.0,
    theta_max: float = 80.0,
    min_cache_files: int = 100,
    rebuild: bool = False,
) -> str:
    """Build the RRUFF processed npz cache once so every fold avoids ASE DB processing."""
    import glob

    from ase.db import connect
    from scipy.interpolate import interp1d

    cache_dir = os.path.join(os.path.dirname(rruff_db_path), "rruff_processed")
    existing = glob.glob(os.path.join(cache_dir, "rruff_*.npz"))
    if not rebuild and len(existing) > min_cache_files:
        print(f"Using cached RRUFF npz files from {cache_dir} ({len(existing)} files).", flush=True)
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Building RRUFF npz cache once at {cache_dir}...", flush=True)
    target_angles = np.linspace(theta_min, theta_max, target_length)
    db = connect(rruff_db_path)
    written = 0
    for row in tqdm(db.select(), desc="Building RRUFF cache", total=db.count(), file=sys.stdout, dynamic_ncols=True):
        data_dict = row.data
        if not data_dict or "angle" not in data_dict or "intensity" not in data_dict:
            continue

        try:
            angles = np.asarray(data_dict["angle"])
            intensities = np.asarray(data_dict["intensity"])
            try:
                intensities = np.clip(intensities - baseline_als(intensities), 0, None)
            except Exception:
                intensities = np.clip(intensities, 0, None)

            f = interp1d(angles, intensities, bounds_error=False, fill_value=0.0)
            y = np.clip(f(target_angles), 0, None).astype(np.float32)
            if y.max() > 0:
                y = y / (y.max() + 1e-8)
            if y.max() <= 0:
                continue

            np.savez_compressed(os.path.join(cache_dir, f"rruff_{row.id}.npz"), y=y)
            written += 1
        except Exception as exc:
            print(f"Skipping RRUFF row {row.id}: {exc}", flush=True)

    print(f"RRUFF cache ready: wrote {written} npz files to {cache_dir}.", flush=True)
    return cache_dir

def build_rruff_reference_bank(dataset, device: torch.device):
    """Build retrieval bank from the current RRUFF fold split."""
    ref_patterns = []
    ref_ids = []
    for phase_id, _rruff_id, intensity in dataset.phases:
        tensor = torch.as_tensor(intensity, dtype=torch.float32)
        if tensor.max().item() > 0:
            tensor = tensor / (tensor.max() + 1e-8)
        ref_patterns.append(tensor)
        ref_ids.append(int(phase_id))

    if not ref_patterns:
        print("RRUFF reference bank is empty; identification metrics disabled.", flush=True)
        return None, None

    return torch.stack(ref_patterns).to(device), torch.tensor(ref_ids, dtype=torch.long, device=device)

def evaluate_fold(model, loader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    ref_bank = ref_ids = None
    if not args.disable_identification:
        ref_bank, ref_ids = build_rruff_reference_bank(loader.dataset, device)

    accumulated = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "pearson_corr": 0.0,
        "sir": 0.0,
        "sar": 0.0,
        "delta_2theta": 0.0,
        "fwhm_error": 0.0,
        **{f"id_acc_top{k}": 0.0 for k in range(1, 11)},
    }
    steps = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc="Testing RRUFF fold", file=sys.stdout, dynamic_ncols=True)
        for batch_idx, batch in enumerate(iterator):
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
            metrics = calculate_separation_metrics(
                aligned_preds,
                targets,
                two_theta_range=(10.0, 80.0),
                calc_detailed=True,
            )

            target_energy = (targets**2).sum(dim=-1)
            target_is_active = (target_energy > 1e-6).float()

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
            accumulated["pearson_corr"] += metrics["pearson_corr"]
            accumulated["sir"] += metrics["sir"]
            accumulated["sar"] += metrics["sar"]
            accumulated["delta_2theta"] += metrics["delta_2theta"]
            accumulated["fwhm_error"] += metrics["fwhm_error"]
            for key, value in id_metrics.items():
                accumulated[key] += value
            steps += 1

    return {key: value / max(1, steps) for key, value in accumulated.items()}

def summarize(metrics: list[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    metadata_keys = {"k", "fold", "num_folds", "virtual_epoch_length"}
    numeric_keys = sorted(
        key
        for key, value in metrics[0].items()
        if key not in metadata_keys and isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    summary = {}
    for key in numeric_keys:
        values = np.asarray([float(item[key]) for item in metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary

def load_model(args: argparse.Namespace):
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = dict(ckpt.get("config", {}))
    baseline_name = args.baseline_name or config.get("baseline_name")
    if baseline_name is None:
        raise ValueError("--baseline_name is required when checkpoint config does not contain it")

    mae_checkpoint = args.mae_checkpoint or os.environ.get("PATH_CKPT_PRETRAIN") or config.get("mae_checkpoint")
    if not mae_checkpoint:
        raise ValueError("MAE checkpoint is required via --mae_checkpoint, PATH_CKPT_PRETRAIN, or checkpoint config")

    num_phases = args.num_phases or int(config.get("num_phases", 4))
    xrd_length = int(config.get("xrd_length", 3500))
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
    return model, config, baseline_name, num_phases, xrd_length

def parse_int_list(values: Iterable[int] | None, default: list[int]) -> list[int]:
    return list(values) if values is not None else default

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Transformer-family baselines on RRUFF k-fold splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_name", choices=["transformer", "itransformer", "patchtst"], default=None)
    parser.add_argument("--mae_checkpoint", type=str, default=None)
    parser.add_argument("--rruff_db", type=str, default=os.environ.get("PATH_DATA_RRUFF"))
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("PATH_OUTPUT_BASELINE_RRUFF_EVAL", "test_results/transformer_family_rruff_kfold"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_phases", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--virtual_epoch_length", type=int, default=1000)
    parser.add_argument("--activity_threshold", type=float, default=0.5)
    parser.add_argument("--disable_identification", action="store_true")
    parser.add_argument("--rebuild_rruff_cache", action="store_true")
    parser.add_argument("--skip_rruff_cache_build", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    model, config, baseline_name, num_phases, xrd_length = load_model(args)
    args.num_phases = num_phases
    if not args.skip_rruff_cache_build:
        ensure_rruff_cache(args.rruff_db, xrd_length, rebuild=args.rebuild_rruff_cache)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device)
    print(f"Model loaded: baseline={baseline_name}, device={device}", flush=True)

    fold_ids = parse_int_list(args.folds, list(range(args.num_folds)))
    all_fold_metrics = []
    per_k_summary = {}

    for k in args.k_values:
        k_metrics = []
        for fold in fold_ids:
            print(f"RRUFF eval: k={k}, fold={fold}/{args.num_folds}", flush=True)
            loader = create_rruff_dataloader(
                args.rruff_db,
                batch_size=args.batch_size,
                min_k=k,
                max_k=k,
                k_weights=[1.0],
                target_length=xrd_length,
                split="test",
                num_folds=args.num_folds,
                fold=fold,
                seed=args.seed,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                virtual_epoch_length=args.virtual_epoch_length,
            )
            print(f"Fold dataloader: {len(loader.dataset)} samples, {len(loader)} batches.", flush=True)
            metrics = evaluate_fold(model, loader, device, args)
            k_metrics.append(metrics)
            all_fold_metrics.append(metrics)

            fold_dir = os.path.join(args.save_dir, f"k{k}", f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)

        per_k_summary[f"k{k}"] = summarize(k_metrics)

    overall_summary = summarize(all_fold_metrics)
    payload = {
        "per_k_mean_std": per_k_summary,
        "overall_mean_std": overall_summary,
    }
    summary_path = os.path.join(args.save_dir, "rruff_kfold_mean_std.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print("RRUFF k-fold summary saved.", flush=True)
    for metric in ("loss", "pearson_corr", "si_sdr", "id_acc_top1", "id_acc_top10"):
        if metric in overall_summary:
            item = overall_summary[metric]
            print(f"overall {metric}: {item['mean']:.4f} ± {item['std']:.4f}", flush=True)

if __name__ == "__main__":
    main()
