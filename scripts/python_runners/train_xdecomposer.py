"""Train XDecomposer."""

import argparse
import logging
import os
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import swanlab as sw
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.config import OnlineMixingConfig
from src.data.online_mixing_dataset import create_online_mixing_dataloader
from src.losses import calculate_pit_loss, calculate_pit_sisdr
from src.models.xdecomposer import build_xdecomposer
from src.models.xrd_transformer import XRDMaskedAutoencoder
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.distributed import cleanup_ddp, setup_ddp
from src.utils.logging import setup_logger
from src.utils.metrics import calculate_peak_position_metrics_batch
from src.utils.optimization import NoamScheduler, get_cosine_schedule_with_warmup
from src.utils.run_outputs import current_timestamp, ensure_timestamp_dir, with_run_timestamp

def visualize_predictions(model, dataset, device, rank, epoch, save_dir, run_timestamp, num_samples=2):
    model.eval()
    if rank != 0:
        return

    dataset_size = len(dataset)
    if dataset_size == 0:
        return
    num_samples = min(dataset_size, num_samples)
    indices = np.random.choice(dataset_size, num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            mix = sample["multiphase_xrd"].unsqueeze(0).to(device)
            targets = sample["single_xrds"].unsqueeze(0).to(device)
            preds, activity_logits = model(mix.unsqueeze(1))
            _, best_perms = calculate_pit_loss(preds, targets)

            perm = best_perms[0].cpu().numpy()
            pred_ordered = preds[0][perm].cpu().numpy()
            act_probs = torch.sigmoid(activity_logits)[0].cpu().numpy()
            targets_np = targets[0].cpu().numpy()
            mix_np = mix[0].cpu().numpy()

            ax_mix = axes[row, 0]
            ax_mix.plot(mix_np, label="Input Mixture", color="black", alpha=0.5, linewidth=2)
            ax_mix.plot(pred_ordered.sum(axis=0), label="Sum of Predictions", color="red", linestyle="--", alpha=0.8)
            ax_mix.set_title(f"Sample {idx}: Mixture Reconstruction")
            ax_mix.legend()

            ax_comp = axes[row, 1]
            cmap = plt.get_cmap("tab10")
            for phase_id, target in enumerate(targets_np):
                color = cmap(phase_id % 10)
                offset = phase_id * 1.1
                if target.max() >= 1e-4:
                    ax_comp.fill_between(
                        range(len(target)),
                        target + offset,
                        offset,
                        color=color,
                        alpha=0.3,
                        label=f"GT {phase_id}",
                    )
                    ax_comp.plot(target + offset, color=color, alpha=0.4, linewidth=1)
                else:
                    ax_comp.text(0, offset + 0.1, f"GT {phase_id} silent", fontsize=8, color=color, alpha=0.6)

                line_style = "-" if act_probs[phase_id] > 0.5 else ":"
                line_width = 1.5 if act_probs[phase_id] > 0.5 else 1.0
                ax_comp.plot(
                    pred_ordered[phase_id] + offset,
                    color=color,
                    linestyle=line_style,
                    linewidth=line_width,
                    label=f"Pred {phase_id} (p={act_probs[phase_id]:.2f})",
                )

            ax_comp.set_title(f"Sample {idx}: Components")
            ax_comp.legend(loc="upper right", fontsize="small", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_epoch_{epoch + 1:04d}_{run_timestamp}.png"))
    plt.close()

def validate_model(model, val_loader, device, rank, lambda_sisdr=0.1):
    model.eval()
    accumulated = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "peak_recall": 0.0,
        "peak_precision": 0.0,
        "peak_f1": 0.0,
        "peak_mean_shift": 0.0,
        "act_acc": 0.0,
        "act_f1": 0.0,
    }
    steps = 0

    with torch.no_grad():
        iterator = tqdm(val_loader, desc="Validating", leave=False) if rank == 0 else val_loader
        for batch in iterator:
            mix = batch["multiphase_xrd"].to(device)
            targets = batch["single_xrds"].to(device)

            with torch.amp.autocast("cuda"):
                preds, activity_logits = model(mix.unsqueeze(1))
                sep_loss, best_perms = calculate_pit_loss(
                    preds,
                    targets,
                    lambda_sisdr=lambda_sisdr,
                    lambda_geo=1.0,
                    mixture_ref=mix,
                    lambda_mix=1.0,
                )

            sisdr = calculate_pit_sisdr(preds, targets)
            target_energy = (targets**2).sum(dim=-1)
            target_is_active = (target_energy > 1e-6).float()
            aligned_activity = torch.gather(target_is_active, 1, best_perms)
            act_pred = (torch.sigmoid(activity_logits) > 0.5).float()

            acc = (act_pred == aligned_activity).float().mean().item()
            tp = (act_pred * aligned_activity).sum()
            fp = (act_pred * (1 - aligned_activity)).sum()
            fn = ((1 - act_pred) * aligned_activity).sum()
            f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

            batch_size, num_phases, sig_len = preds.shape
            peak_metrics = calculate_peak_position_metrics_batch(
                preds.reshape(batch_size * num_phases, sig_len),
                targets.reshape(batch_size * num_phases, sig_len),
            )

            accumulated["loss"] += sep_loss.item()
            accumulated["si_sdr"] += sisdr
            accumulated["peak_recall"] += peak_metrics["peak_recall"]
            accumulated["peak_precision"] += peak_metrics["peak_precision"]
            accumulated["peak_f1"] += peak_metrics["peak_f1"]
            accumulated["peak_mean_shift"] += peak_metrics["peak_mean_shift"]
            accumulated["act_acc"] += acc
            accumulated["act_f1"] += f1
            steps += 1

    return {key: value / max(1, steps) for key, value in accumulated.items()}

def resolve_num_workers(requested_workers: int, world_size: int) -> int:
    if requested_workers > 0:
        return requested_workers

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 1

    per_rank_budget = max(1, available_cpus // max(1, world_size))
    return max(1, min(8, per_rank_budget))

def parse_args():
    parser = argparse.ArgumentParser(description="Train XDecomposer")
    parser.add_argument("--mae_checkpoint", type=str, required=True)
    parser.add_argument("--singlephase_xrd_db", type=str, required=True)
    parser.add_argument("--crystal_db", type=str, default="")
    parser.add_argument("--xrd_length", type=int, default=3500)
    parser.add_argument("--num_phases", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cnn_channels", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--cnn_kernels", type=int, nargs="+", default=[15, 8, 8, 10])
    parser.add_argument("--cnn_strides", type=int, nargs="+", default=[1, 2, 2, 5])
    parser.add_argument("--no_transformer", action="store_true", help="Disable the transformer bottleneck")
    parser.add_argument("--no_film", action="store_true", help="Disable FiLM modulation")
    parser.add_argument("--no_skip_connections", action="store_true", help="Disable decoder skip connections")
    parser.add_argument("--mask_type", type=str, default="soft", choices=["soft", "hard", "direct"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.environ.get("PATH_OUTPUT_XDECOMPOSER_ROOT", "./checkpoints/xdecomposer"),
    )
    parser.add_argument("--experiment_name", type=str, default="XDecomposer")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--lambda_sisdr", type=float, default=0.1)
    parser.add_argument("--lambda_activity", type=float, default=2.0)
    parser.add_argument("--lambda_geo", type=float, default=1.0)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--activity_threshold", type=float, default=0.8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_scheduler", type=str, default="noam", choices=["cosine", "noam"])
    parser.add_argument("--noam_factor", type=float, default=1.0)
    parser.add_argument("--vis_interval", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def apply_ablation_config(args):
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config.get("model_ablation", {})
    if "use_transformer" in model_cfg:
        args.no_transformer = not model_cfg["use_transformer"]
    if "use_film" in model_cfg:
        args.no_film = not model_cfg["use_film"]
    if "use_skip_connections" in model_cfg:
        args.no_skip_connections = not model_cfg["use_skip_connections"]
    if "mask_type" in model_cfg:
        args.mask_type = model_cfg["mask_type"]

    train_cfg = config.get("training_ablation", {})
    if "lambda_geo" in train_cfg:
        args.lambda_geo = train_cfg["lambda_geo"]

    return args

def prepare_output_dir(args):
    if args.resume:
        args.save_dir = os.path.dirname(os.path.abspath(args.resume))
        args.run_timestamp = current_timestamp()
        return args

    args.save_dir, args.run_timestamp = ensure_timestamp_dir(args.save_dir)
    return args

def main():
    args = prepare_output_dir(apply_ablation_config(parse_args()))

    env_rank = int(os.environ.get("RANK", "0")) if "WORLD_SIZE" in os.environ else 0
    if env_rank == 0:
        setup_logger(args.save_dir, 0)
        logging.info(
            "Launcher start: save_dir=%s experiment=%s config=%s",
            args.save_dir,
            args.experiment_name,
            args.config,
        )

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    num_workers = resolve_num_workers(args.num_workers, world_size)

    data_config = OnlineMixingConfig(
        XRD_LENGTH=args.xrd_length,
        MIN_K=2,
        MAX_K=args.num_phases,
        AUGMENT=args.augment,
    )

    if rank == 0:
        setup_logger(args.save_dir, rank)
        logging.info(
            "DDP initialized: rank=%d local_rank=%d world_size=%d run_timestamp=%s",
            rank,
            local_rank,
            world_size,
            args.run_timestamp,
        )
        if args.config:
            logging.info("Loaded ablation config from %s", args.config)

    train_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db,
        args.crystal_db,
        data_config,
        split="train",
        batch_size=args.batch_size,
        num_workers=num_workers,
        distributed=True,
    )
    val_loader = create_online_mixing_dataloader(
        args.singlephase_xrd_db,
        args.crystal_db,
        data_config,
        split="val",
        batch_size=args.batch_size,
        num_workers=num_workers,
        distributed=True,
    )

    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    if train_dataset_size == 0:
        raise ValueError(
            f"No training samples found under --singlephase_xrd_db={args.singlephase_xrd_db}. "
            "Check PATH_DATA_SINGLEPHASE and ensure the directory contains valid .npz files."
        )
    if val_dataset_size == 0:
        raise ValueError(
            f"No validation samples found under --singlephase_xrd_db={args.singlephase_xrd_db}. "
            "Add more data or adjust the dataset split."
        )
    if train_steps == 0:
        raise ValueError(
            f"Training loader is empty: train_size={train_dataset_size}, batch_size={args.batch_size}, "
            f"world_size={world_size}. Reduce batch size or add more data."
        )
    if val_steps == 0:
        raise ValueError(
            f"Validation loader is empty: val_size={val_dataset_size}, batch_size={args.batch_size}. "
            "Reduce batch size or add more data."
        )

    if rank == 0:
        logging.info(
            "Data loader workers per rank: %d (requested=%d, world_size=%d)",
            num_workers,
            args.num_workers,
            world_size,
        )
        swan_config = with_run_timestamp(vars(args), args.run_timestamp)
        swan_config.update(asdict(data_config))
        swan_project = os.getenv("SWANLAB_PROJECT", "XDecomposer")
        sw.init(
            project=swan_project,
            experiment_name=f"{args.experiment_name}-{args.run_timestamp}",
            config=swan_config,
        )

    if rank == 0:
        logging.info("Loading encoder checkpoint from %s", args.mae_checkpoint)

    mae_ckpt = torch.load(args.mae_checkpoint, map_location="cpu")
    mae_config = mae_ckpt.get("config", {})
    mae = XRDMaskedAutoencoder(
        xrd_length=args.xrd_length,
        d_model=mae_config.get("d_model", 768),
        n_layers=mae_config.get("n_layers", 4),
        n_heads=mae_config.get("n_heads", 12),
        decoder_d_model=mae_config.get("decoder_dim", 512),
        decoder_n_layers=mae_config.get("decoder_layers", 4),
    )
    mae_state = mae_ckpt["model_state_dict"] if "model_state_dict" in mae_ckpt else mae_ckpt
    mae.load_state_dict(mae_state)
    del mae_ckpt

    model = build_xdecomposer(
        mae,
        num_sources=args.num_phases,
        cnn_channels=args.cnn_channels,
        cnn_kernels=args.cnn_kernels,
        cnn_strides=args.cnn_strides,
        use_transformer=not args.no_transformer,
        use_film=not args.no_film,
        use_skip_connections=not args.no_skip_connections,
        mask_type=args.mask_type,
    ).to(device)
    needs_unused_parameter_detection = args.no_transformer or args.no_film
    if rank == 0:
        logging.info(
            "DDP find_unused_parameters=%s (no_transformer=%s, no_film=%s)",
            needs_unused_parameter_detection,
            args.no_transformer,
            args.no_film,
        )
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=needs_unused_parameter_detection,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch = train_steps
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, args.warmup_epochs * steps_per_epoch)

    if args.lr_scheduler == "noam":
        scheduler = NoamScheduler(optimizer, d_model=mae.d_model, warmup_steps=warmup_steps, factor=args.noam_factor)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, steps_per_epoch * args.epochs)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        if rank == 0:
            logging.info("Resuming from %s", args.resume)
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device, weights_only=False)

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        steps = 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}") if rank == 0 else train_loader

        for batch in iterator:
            mix = batch["multiphase_xrd"].to(device)
            targets = batch["single_xrds"].to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds, activity_logits = model(mix.unsqueeze(1))
                sep_loss, best_perms = calculate_pit_loss(
                    preds,
                    targets,
                    alpha=args.alpha,
                    lambda_sisdr=args.lambda_sisdr,
                    lambda_geo=args.lambda_geo,
                    mixture_ref=mix,
                    lambda_mix=args.lambda_mix,
                )
                target_energy = (targets**2).sum(dim=-1)
                target_is_active = (target_energy > 1e-6).float()
                aligned_activity = torch.gather(target_is_active, 1, best_perms)
                activity_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    activity_logits,
                    aligned_activity,
                )
                loss = sep_loss + args.lambda_activity * activity_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            steps += 1
            if rank == 0:
                act_pred = (torch.sigmoid(activity_logits) > 0.5).float()
                act_acc = (act_pred == aligned_activity).float().mean().item()
                iterator.set_postfix(loss=f"{loss.item():.4f}", act_acc=f"{act_acc:.2f}")
                if steps % args.log_interval == 0:
                    sw.log(
                        {
                            "train/loss_step": loss.item(),
                            "train/act_loss": activity_loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }
                    )

        val_metrics = validate_model(model, val_loader, device, rank, lambda_sisdr=args.lambda_sisdr)

        if rank == 0:
            logging.info(
                "Epoch %d | Train %.5f | Val %.5f | SI-SDR %.2f | Peak-F1 %.3f | Act-Acc %.2f",
                epoch + 1,
                epoch_loss / max(1, steps),
                val_metrics["loss"],
                val_metrics["si_sdr"],
                val_metrics["peak_f1"],
                val_metrics["act_acc"],
            )
            sw.log(
                {
                    "train/loss": epoch_loss / max(1, steps),
                    "val/loss": val_metrics["loss"],
                    "val/act_acc": val_metrics["act_acc"],
                    "val/act_f1": val_metrics["act_f1"],
                    "val/si_sdr": val_metrics["si_sdr"],
                    "val/peak_recall": val_metrics["peak_recall"],
                    "val/peak_precision": val_metrics["peak_precision"],
                    "val/peak_f1": val_metrics["peak_f1"],
                    "val/peak_mean_shift": val_metrics["peak_mean_shift"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

            checkpoint_config = with_run_timestamp(vars(args), args.run_timestamp)
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    os.path.join(args.save_dir, "best.pt"),
                    checkpoint_config,
                    rank,
                    verbose=False,
                )
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                os.path.join(args.save_dir, "latest.pt"),
                checkpoint_config,
                rank,
                verbose=False,
            )

            if (epoch + 1) % args.vis_interval == 0:
                visualize_predictions(
                    model,
                    val_loader.dataset,
                    device,
                    rank,
                    epoch,
                    args.save_dir,
                    args.run_timestamp,
                )

    cleanup_ddp()

if __name__ == "__main__":
    main()
