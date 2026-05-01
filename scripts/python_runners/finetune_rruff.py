"""RRUFF per-fold fine-tuning for 5-fold cross-validation."""

import argparse, copy, json, logging, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.core import XRDCollateFunction
from src.data.rruff_dataset import RRUFFOnlineMixingDataset
from src.losses import calculate_pit_loss
from src.models.xdecomposer import build_xdecomposer
from src.models.xrd_transformer import XRDMaskedAutoencoder
from src.utils.metrics import calculate_separation_metrics
from src.utils.optimization import get_cosine_schedule_with_warmup
from src.utils.run_outputs import current_timestamp

def build_model(checkpoint_path, device):
    """Load pre-trained XDecomposer from MP20 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    mae_ckpt_path = os.environ.get("PATH_CKPT_PRETRAIN") or config.get("mae_checkpoint", "")
    mae_ckpt = torch.load(mae_ckpt_path, map_location="cpu")
    mc = mae_ckpt.get("config", {})

    mae = XRDMaskedAutoencoder(
        xrd_length=config["xrd_length"],
        d_model=mc.get("d_model", 768), n_layers=mc.get("n_layers", 4),
        n_heads=mc.get("n_heads", 12), decoder_d_model=mc.get("decoder_dim", 512),
        decoder_n_layers=mc.get("decoder_layers", 4),
    )
    model = build_xdecomposer(
        mae, num_sources=config["num_phases"],
        cnn_channels=config.get("cnn_channels", [64, 128, 256, 512]),
        cnn_kernels=config.get("cnn_kernels"), cnn_strides=config.get("cnn_strides"),
        use_transformer=not config.get("no_transformer", False),
        use_film=not config.get("no_film", False),
        use_skip_connections=not config.get("no_skip_connections", False),
        mask_type=config.get("mask_type", "soft"),
    )
    state = {k.removeprefix("module."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    return model.to(device), config

def create_datasets(args):
    """Create train / val / test RRUFF datasets for one fold."""
    common = dict(
        rruff_db_path=args.data_dir, min_k=args.min_k, max_k=args.max_k,
        k_weights=args.k_weights, num_folds=args.num_folds, fold=args.fold, seed=args.seed,
    )

    # Train split (4/5) → further 85/15 for finetune/val
    train_full = RRUFFOnlineMixingDataset(**common, split="train", virtual_epoch_length=args.virtual_epoch_length)
    phases = list(train_full.phases)
    np.random.RandomState(args.seed + args.fold).shuffle(phases)
    n_val = max(1, int(len(phases) * 0.15))

    train_full.phases = phases[n_val:]
    val_ds = RRUFFOnlineMixingDataset(**common, split="train", virtual_epoch_length=max(256, args.virtual_epoch_length // 4))
    val_ds.phases = phases[:n_val]

    # Test split (1/5)
    test_ds = RRUFFOnlineMixingDataset(**common, split="test", virtual_epoch_length=args.virtual_epoch_length)

    logging.info("Phases — train: %d, val: %d, test: %d", len(train_full.phases), len(val_ds.phases), len(test_ds.phases))
    return train_full, val_ds, test_ds

def compute_loss(model, batch, device, args):
    """Shared forward + loss for train/val."""
    mix = batch["multiphase_xrd"].to(device)
    targets = batch["single_xrds"].to(device)

    with torch.amp.autocast("cuda"):
        preds, activity_logits = model(mix.unsqueeze(1))
        sep_loss, best_perms = calculate_pit_loss(
            preds, targets, alpha=args.alpha, lambda_sisdr=args.lambda_sisdr,
            lambda_geo=args.lambda_geo, mixture_ref=mix, lambda_mix=args.lambda_mix,
        )
        target_is_active = ((targets ** 2).sum(dim=-1) > 1e-6).float()
        aligned_activity = torch.gather(target_is_active, 1, best_perms)
        act_loss = F.binary_cross_entropy_with_logits(activity_logits, aligned_activity)
        loss = sep_loss + args.lambda_activity * act_loss

    return loss

def train_one_epoch(model, loader, optimizer, scaler, scheduler, device, args):
    model.train()
    total, steps = 0.0, 0
    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()
        loss = compute_loss(model, batch, device, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        total += loss.item(); steps += 1
    return total / max(1, steps)

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total, steps = 0.0, 0
    for batch in tqdm(loader, desc="  val", leave=False):
        total += compute_loss(model, batch, device, args).item(); steps += 1
    return total / max(1, steps)

@torch.no_grad()
def evaluate_test(model, loader, device):
    """Full test evaluation with separation + identification metrics."""
    model.eval()
    ds = loader.dataset
    ref_bank = torch.from_numpy(np.stack([p[2] for p in ds.phases])).to(device)
    ref_ids = torch.from_numpy(np.array([p[0] for p in ds.phases], dtype=np.int64)).to(device)
    ref_norm = F.normalize(ref_bank, p=2, dim=1)

    acc = {"loss": 0., "si_sdr": 0., "pearson_corr": 0., "sir": 0., "sar": 0., "delta_2theta": 0., "fwhm_error": 0.}
    for k in range(1, 11): acc[f"id_acc_top{k}"] = 0.
    steps = 0

    for batch in tqdm(loader, desc="  test"):
        mix = batch["multiphase_xrd"].to(device)
        targets = batch["single_xrds"].to(device)
        phase_ids = batch["phase_ids"].to(device)
        preds, _ = model(mix.unsqueeze(1))

        if targets.shape[1] < preds.shape[1]:
            d = preds.shape[1] - targets.shape[1]
            targets = torch.cat([targets, torch.zeros(targets.shape[0], d, targets.shape[2], device=device)], 1)
            phase_ids = torch.cat([phase_ids, torch.full((phase_ids.shape[0], d), -1, device=device, dtype=torch.long)], 1)

        sep_loss, best_perms = calculate_pit_loss(preds, targets)
        B, K, L = preds.shape
        inv = torch.argsort(best_perms, dim=1)
        aligned = torch.gather(preds, 1, inv.unsqueeze(-1).expand(-1, -1, L))

        # Identification
        sim = torch.matmul(F.normalize(preds.view(-1, L), p=2, dim=1), ref_norm.T)
        _, top_idx = torch.topk(sim, k=min(10, ref_bank.shape[0]), dim=1)
        topk_ids = ref_ids[top_idx]
        aligned_topk = torch.gather(topk_ids.view(B, K, -1), 1, inv.unsqueeze(-1).expand(-1, -1, topk_ids.shape[-1]))
        gt_flat = phase_ids.view(-1, 1)
        active_flat = ((targets ** 2).sum(-1) > 1e-4).float().view(-1)
        cum = torch.cumsum((aligned_topk.view(-1, aligned_topk.shape[-1]) == gt_flat).float(), dim=1)
        n_active = active_flat.sum().item()
        for ki in range(min(10, cum.shape[1])):
            acc[f"id_acc_top{ki+1}"] += ((cum[:, ki] > 0).float() * active_flat).sum().item() / max(1, n_active) if n_active > 0 else 1.0

        m = calculate_separation_metrics(aligned, targets, two_theta_range=(5., 90.), calc_detailed=True)
        acc["loss"] += sep_loss.item()
        for k_ in ("si_sdr", "pearson_corr", "sir", "sar", "delta_2theta", "fwhm_error"):
            acc[k_] += m[k_]
        steps += 1

    return {k: v / max(1, steps) for k, v in acc.items()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", default="test_results/rruff_finetune_kfold")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_k", type=int, default=2)
    p.add_argument("--max_k", type=int, default=4)
    p.add_argument("--k_weights", type=float, nargs="+", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--virtual_epoch_length", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--lambda_sisdr", type=float, default=0.5)
    p.add_argument("--lambda_geo", type=float, default=5.0)
    p.add_argument("--lambda_mix", type=float, default=5.0)
    p.add_argument("--lambda_activity", type=float, default=2.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    ts = current_timestamp()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.save_dir, f"finetune_{ts}.log"))], force=True)

    logging.info("RRUFF Fine-Tuning — Fold %d/%d", args.fold, args.num_folds)
    device = torch.device(args.device)

    model, config = build_model(args.checkpoint, device)
    train_ds, val_ds, test_ds = create_datasets(args)
    collate = XRDCollateFunction()
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=2, pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=2, pin_memory=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, 5 * len(train_loader)), max(1, args.epochs * len(train_loader)))

    best_val, patience_cnt, best_state = float("inf"), 0, None
    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, args)
        v_loss = validate(model, val_loader, device, args)
        logging.info("Epoch %3d | train %.5f | val %.5f | lr %.2e", epoch + 1, t_loss, v_loss, optimizer.param_groups[0]["lr"])

        if v_loss < best_val:
            best_val, patience_cnt = v_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model_state_dict": best_state, "epoch": epoch, "config": config, "fold": args.fold},
                       os.path.join(args.save_dir, "best_finetuned.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logging.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state: model.load_state_dict(best_state)
    metrics = evaluate_test(model, test_loader, device)

    logging.info("Test Results — Fold %d:", args.fold)
    for k, v in metrics.items(): logging.info("  %s: %.4f", k, v)

    out = {**metrics, "fold": args.fold, "run_timestamp": ts, "best_val_loss": best_val}
    for p in (os.path.join(args.save_dir, "test_metrics.json"), os.path.join(args.save_dir, f"test_metrics_{ts}.json")):
        with open(p, "w") as f: json.dump(out, f, indent=4)

if __name__ == "__main__":
    main()
