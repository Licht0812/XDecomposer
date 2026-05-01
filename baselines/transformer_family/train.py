"""DDP training runner for Transformer XRD separation baselines."""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'codex')}"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import swanlab as sw
except ImportError as exc:  # pragma: no cover - handled at runtime
    sw = None
    _SWANLAB_IMPORT_ERROR = exc
else:
    _SWANLAB_IMPORT_ERROR = None

from baselines.transformer_family.models import build_transformer_family_baseline
from baselines.transformer_family.losses import calculate_baseline_loss, calculate_pit_loss, calculate_pit_sisdr
from src.data.config import OnlineMixingConfig
from src.data.online_mixing_dataset import create_online_mixing_dataloader
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.distributed import cleanup_ddp, setup_ddp
from src.utils.logging import setup_logger
from src.utils.metrics import calculate_separation_metrics
from src.utils.optimization import NoamScheduler, get_cosine_schedule_with_warmup

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_config.yaml")
TRANSFORMER_MIXTURE_LOSS_WEIGHT = 5.0

def _flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the YAML sections into argparse-compatible defaults."""
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat

def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value

def _load_yaml_defaults(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return _flatten_config(_expand_env_vars(config))

def _format_template(template: str, args: argparse.Namespace, timestamp: str) -> str:
    values = vars(args).copy()
    values["timestamp"] = timestamp
    return template.format(**values)

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def reduce_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_distributed():
        return metrics
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(keys, values)}

def limit_steps(loader_len: int, max_steps: int | None) -> int:
    if max_steps is None or max_steps <= 0:
        return max(1, loader_len)
    return max(1, min(loader_len, max_steps))

def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()

def mixture_loss_weight(baseline_name: str) -> float:
    """Soft mixture consistency is only needed by the direct Transformer decoder."""
    return TRANSFORMER_MIXTURE_LOSS_WEIGHT if baseline_name == "transformer" else 0.0

def align_predictions(preds: torch.Tensor, best_perms: torch.Tensor) -> torch.Tensor:
    batch_size, num_sources, length = preds.shape
    inv_perms = torch.argsort(best_perms, dim=1)
    return torch.gather(preds, 1, inv_perms.unsqueeze(-1).expand(batch_size, num_sources, length))

def validate_model(model, val_loader, device, args, rank: int) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "pearson_corr": 0.0,
    }
    steps = 0

    iterator = tqdm(val_loader, desc="Validating", leave=False, file=sys.stdout) if rank == 0 else val_loader
    mix_weight = mixture_loss_weight(args.baseline_name)
    with torch.no_grad():
        for batch in iterator:
            if args.max_val_steps is not None and args.max_val_steps > 0 and steps >= args.max_val_steps:
                break
            mix = batch["multiphase_xrd"].to(device, non_blocking=True)
            targets = batch["single_xrds"].to(device, non_blocking=True)

            with autocast_context(device):
                preds, activity_logits = model(mix)
                loss, best_perms, loss_parts, _aligned_activity = calculate_baseline_loss(
                    preds,
                    targets,
                    activity_logits,
                    lambda_activity=args.lambda_activity,
                    mixture_ref=mix if mix_weight > 0 else None,
                    lambda_mix=mix_weight,
                )

            aligned_preds = align_predictions(preds, best_perms)
            sep_metrics = calculate_separation_metrics(
                aligned_preds,
                targets,
                two_theta_range=(5.0, 90.0),
                calc_detailed=False,
            )
            sisdr = calculate_pit_sisdr(preds, targets)

            totals["loss"] += loss.item()
            totals["si_sdr"] += sisdr
            totals["pearson_corr"] += sep_metrics["pearson_corr"]
            steps += 1

    local = {k: v / max(1, steps) for k, v in totals.items()}
    return reduce_metrics(local, device)

def visualize_separation(model, dataset, device, rank, epoch: int, save_dir: str, num_samples: int = 2) -> None:
    if rank != 0:
        return
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(18, 4 * num_samples))
    if num_samples == 1:
        axes = np.asarray(axes).reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[int(idx)]
            mix = sample["multiphase_xrd"].unsqueeze(0).to(device)
            targets = sample["single_xrds"].unsqueeze(0).to(device)

            with autocast_context(device):
                preds, activity_logits = model(mix)
                _, best_perms = calculate_pit_loss(preds, targets)

            pred_ordered = align_predictions(preds, best_perms)[0].float().cpu().numpy()
            targets_np = targets[0].float().cpu().numpy()
            mix_np = mix[0].float().cpu().numpy()
            act_probs = torch.sigmoid(activity_logits)[0].float().cpu().numpy()

            ax_mix = axes[row, 0]
            ax_mix.plot(mix_np, label="Mixture", color="black", linewidth=1.5, alpha=0.7)
            ax_mix.plot(pred_ordered.sum(axis=0), label="Pred Sum", color="red", linewidth=1.2, alpha=0.8)
            ax_mix.set_title(f"Sample {idx}: mixture")
            ax_mix.legend(fontsize="small")

            ax_src = axes[row, 1]
            cmap = plt.get_cmap("tab10")
            for k in range(targets_np.shape[0]):
                offset = k * 1.1
                color = cmap(k % 10)
                if targets_np[k].max() > 1e-4:
                    ax_src.plot(targets_np[k] + offset, color=color, linewidth=1.0, alpha=0.45, label=f"GT {k}")
                ax_src.plot(
                    pred_ordered[k] + offset,
                    color=color,
                    linewidth=1.2,
                    linestyle="-" if act_probs[k] > 0.5 else ":",
                    label=f"Pred {k} p={act_probs[k]:.2f}",
                )
            ax_src.set_title("Separated sources")
            ax_src.legend(fontsize="x-small", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_epoch_{epoch + 1}.png"), dpi=140)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args()
    yaml_defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Train Transformer XRD separation baselines")

    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--baseline_name", choices=["transformer", "itransformer", "patchtst"])
    parser.add_argument("--mae_checkpoint", type=str)

    parser.add_argument("--singlephase_xrd_db", type=str)
    parser.add_argument("--crystal_db", type=str)
    parser.add_argument("--xrd_length", type=int, default=3500)
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=None)
    parser.add_argument("--k_weights", type=float, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--noise_level", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch_len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--output_activation", choices=["relu", "softplus", "none"], default="relu")

    parser.add_argument("--transformer_decoder_layers", type=int, default=4)
    parser.add_argument("--transformer_d_ff", type=int, default=None)
    parser.add_argument("--itransformer_dim", type=int, default=256)
    parser.add_argument("--itransformer_layers", type=int, default=2)
    parser.add_argument("--itransformer_heads", type=int, default=8)
    parser.add_argument("--itransformer_d_ff", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--save_dir_template",
        type=str,
        default=os.environ.get(
            "PATH_TEMPLATE_BASELINE_SAVE_DIR",
            "checkpoints/transformer_{baseline_name}_{run_name}_{timestamp}",
        ),
    )
    parser.add_argument("--run_name", type=str, default="separation")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--experiment_name_template", type=str, default="{baseline_name}_{run_name}")
    parser.add_argument("--lambda_activity", type=float, default=2.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr_scheduler", choices=["cosine", "noam"], default="cosine")
    parser.add_argument("--noam_factor", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--vis_interval", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)

    parser.add_argument("--swanlab_project", type=str, default=os.getenv("SWANLAB_PROJECT", "XRD-Transformer-Baselines"))
    parser.add_argument("--disable_swanlab", action="store_true")
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()

    if not args.baseline_name:
        parser.error("--baseline_name is required either in YAML config or CLI")
    if not args.mae_checkpoint:
        parser.error("--mae_checkpoint is required either in YAML config or CLI")
    if not args.singlephase_xrd_db:
        parser.error("--singlephase_xrd_db is required either in YAML config or CLI")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.experiment_name:
        args.experiment_name = _format_template(args.experiment_name_template, args, timestamp)
    if not args.save_dir:
        args.save_dir = _format_template(args.save_dir_template, args, timestamp)
    return args

def main() -> None:
    args = parse_args()
    env_rank = int(os.environ.get("RANK", "0")) if "WORLD_SIZE" in os.environ else 0
    if env_rank == 0:
        setup_logger(args.save_dir, 0)
        logging.info("Launcher start: baseline=%s, save_dir=%s", args.baseline_name, args.save_dir)

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    max_k = args.max_k if args.max_k is not None else args.num_phases
    data_kwargs = {
        "XRD_LENGTH": args.xrd_length,
        "MIN_K": args.min_k,
        "MAX_K": max_k,
        "AUGMENT": args.augment,
        "NOISE_LEVEL": args.noise_level,
        "SEED": args.seed,
    }
    if args.k_weights is not None:
        data_kwargs["K_WEIGHTS"] = tuple(args.k_weights)
        data_kwargs["K_DISTRIBUTION"] = "weighted"
    data_config = OnlineMixingConfig(**data_kwargs)

    if rank == 0:
        setup_logger(args.save_dir, rank)
        logging.info("DDP initialized: rank=%d, local_rank=%d, world_size=%d", rank, local_rank, world_size)
        if not args.disable_swanlab:
            if sw is None:
                raise RuntimeError("swanlab is required unless --disable_swanlab is set") from _SWANLAB_IMPORT_ERROR
            swanlab_api_key = os.getenv("SWANLAB_API_KEY")
            if swanlab_api_key:
                sw.login(api_key=swanlab_api_key)
            swan_config = vars(args).copy()
            swan_config.update(asdict(data_config))
            sw.init(project=args.swanlab_project, experiment_name=args.experiment_name, config=swan_config)

    train_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db,
        args.crystal_db,
        data_config,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        distributed=is_distributed(),
    )
    val_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db,
        args.crystal_db,
        data_config,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        distributed=is_distributed(),
    )

    model = build_transformer_family_baseline(
        name=args.baseline_name,
        mae_checkpoint=args.mae_checkpoint,
        num_sources=args.num_phases,
        xrd_length=args.xrd_length,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        output_activation=args.output_activation,
        transformer_decoder_layers=args.transformer_decoder_layers,
        transformer_d_ff=args.transformer_d_ff,
        itransformer_dim=args.itransformer_dim,
        itransformer_layers=args.itransformer_layers,
        itransformer_heads=args.itransformer_heads,
        itransformer_d_ff=args.itransformer_d_ff,
    ).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    steps_per_epoch = limit_steps(len(train_loader), args.max_train_steps)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, args.warmup_epochs * steps_per_epoch)
    if args.lr_scheduler == "noam":
        d_model_for_noam = args.d_model or getattr(model.module if hasattr(model, "module") else model, "d_model", 768)
        scheduler = NoamScheduler(optimizer, d_model=d_model_for_noam, warmup_steps=warmup_steps, factor=args.noam_factor)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps_per_epoch * args.epochs)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        if rank == 0:
            logging.info("Resuming baseline training from %s", args.resume)
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device, weights_only=False)
    elif args.finetune:
        if rank == 0:
            logging.info("Finetuning baseline weights from %s", args.finetune)
        load_checkpoint(args.finetune, model, optimizer, scheduler, device, weights_only=True)

    if rank == 0:
        logging.info(
            "Training steps per epoch: %d%s | Validation max steps: %s",
            steps_per_epoch,
            " (limited)" if args.max_train_steps is not None and args.max_train_steps > 0 else "",
            args.max_val_steps if args.max_val_steps is not None and args.max_val_steps > 0 else "all",
        )
        logging.info("DataLoader: num_workers=%d, pin_memory=%s", args.num_workers, args.pin_memory)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        steps = 0
        mix_weight = mixture_loss_weight(args.baseline_name)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}", file=sys.stdout) if rank == 0 else train_loader
        for batch in iterator:
            if args.max_train_steps is not None and args.max_train_steps > 0 and steps >= args.max_train_steps:
                break
            mix = batch["multiphase_xrd"].to(device, non_blocking=True)
            targets = batch["single_xrds"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                preds, activity_logits = model(mix)
                loss, best_perms, loss_parts, _aligned_activity = calculate_baseline_loss(
                    preds,
                    targets,
                    activity_logits,
                    lambda_activity=args.lambda_activity,
                    mixture_ref=mix if mix_weight > 0 else None,
                    lambda_mix=mix_weight,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.detach().item()
            steps += 1

            if rank == 0:
                iterator.set_postfix(loss=f"{loss.item():.4f}")
                if not args.disable_swanlab:
                    log_payload = {
                        "train/loss_step": loss.item(),
                        "train/sep_loss_step": loss_parts["sep_loss"].item(),
                        "train/activity_loss_step": loss_parts["activity_loss"].item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    if "mixture_loss" in loss_parts:
                        log_payload["train/mixture_loss_step"] = loss_parts["mixture_loss"].item()
                    sw.log(log_payload)

        local_train = {
            "train_loss": epoch_loss / max(1, steps),
        }
        train_metrics = reduce_metrics(local_train, device)
        val_metrics = validate_model(model, val_loader, device, args, rank)

        if rank == 0:
            logging.info(
                "Epoch %d | Train %.5f | Val %.5f | SI-SDR %.3f | Pearson %.3f",
                epoch + 1,
                train_metrics["train_loss"],
                val_metrics["loss"],
                val_metrics["si_sdr"],
                val_metrics["pearson_corr"],
            )
            if not args.disable_swanlab:
                log_payload = {
                    "train/loss": train_metrics["train_loss"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                sw.log(log_payload)

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.save_dir, "best.pt"), vars(args), rank)
            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(args.save_dir, "latest.pt"), vars(args), rank)

            if args.vis_interval > 0 and (epoch + 1) % args.vis_interval == 0:
                visualize_separation(model, val_loader.dataset, device, rank, epoch, args.save_dir)

    if rank == 0 and not args.disable_swanlab and sw is not None:
        sw.finish()
    cleanup_ddp()

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp()
