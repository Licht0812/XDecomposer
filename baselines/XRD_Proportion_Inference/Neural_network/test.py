import argparse
import os
import time

import torch

from data_utils import OnlineMixingConfig, get_dataloaders
from metrics_utils import SeparationLoss, calculate_all_metrics
from model import get_model


config = OnlineMixingConfig(
    MIN_K=2,
    MAX_K=4,
    MIN_WEIGHT=0.15,
    XRD_LENGTH=3500,
    AUGMENT=False,
    NOISE_LEVEL=0.01,
    SEED=7,
)

DB_PATH = "data/mp20-xrd_data"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_separation_model.pth"
NUM_WORKERS = 8


def build_reference_library(test_loader):
    """Build a reference library from the test split."""
    print("Building reference library...")
    library = {}

    for batch in test_loader:
        xrds = batch["single_xrds"]
        phase_ids = batch["phase_ids"]
        weights = batch["weights"]

        batch_size, num_sources, _ = xrds.shape
        for batch_idx in range(batch_size):
            for source_idx in range(num_sources):
                phase_id = int(phase_ids[batch_idx, source_idx])
                weight = float(weights[batch_idx, source_idx])
                if phase_id != -1 and weight > 1e-6 and phase_id not in library:
                    library[phase_id] = (xrds[batch_idx, source_idx] / weight).cpu()

    ref_ids = sorted(library.keys())
    ref_patterns = torch.stack([library[ref_id] for ref_id in ref_ids])
    print(f"Reference library size: {len(ref_ids)}")
    return ref_patterns.to(DEVICE), ref_ids


def evaluate_model(model, test_loader, ref_lib, ref_ids, num_phases):
    """Evaluate the model on the test split."""
    model.eval()
    criterion = SeparationLoss()
    running_metrics = {
        "loss": 0.0,
        "si_sdr": 0.0,
        "pearson_corr": 0.0,
        "sir": 0.0,
        "sar": 0.0,
        "delta_2theta": 0.0,
        "fwhm_error": 0.0,
        **{f"id_acc_top{k}": 0.0 for k in range(1, 11)},
    }
    sample_count = 0
    phase_count = 0

    print(f"Evaluating {num_phases} phases...")
    start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["multiphase_xrd"].to(DEVICE)
            targets = batch["single_xrds"].to(DEVICE)
            phase_ids = batch["phase_ids"].to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_metrics = calculate_all_metrics(
                outputs,
                targets,
                phase_ids,
                reference_library=ref_lib,
                reference_ids=ref_ids,
            )
            active_mask = torch.sum(targets ** 2, dim=-1) > 1e-4
            batch_phase_count = int(active_mask.sum().item())

            running_metrics["loss"] += loss.item() * inputs.size(0)
            for key in batch_metrics:
                if key in running_metrics:
                    running_metrics[key] += batch_metrics[key] * batch_phase_count
            sample_count += inputs.size(0)
            phase_count += batch_phase_count

    elapsed = time.time() - start
    test_metrics = {}
    for key, value in running_metrics.items():
        denom = sample_count if key == "loss" else max(1, phase_count)
        test_metrics[key] = value / max(1, denom)

    print(f"Finished in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"--- Results for {num_phases} phases ---")
    for key, value in test_metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the separation baseline.")
    parser.add_argument(
        "--num_phases",
        type=int,
        default=0,
        help="Set the number of phases to 2, 3, or 4. Use 0 for a mixed test split.",
    )
    args = parser.parse_args()

    if args.num_phases in [2, 3, 4]:
        print(f"Using fixed {args.num_phases}-phase mixtures.")
        config.MIN_K = args.num_phases
        config.MAX_K = args.num_phases
    else:
        print("Using mixed 2-4 phase mixtures.")

    _, _, test_loader = get_dataloaders(
        DB_PATH,
        config,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    ref_lib, ref_ids = build_reference_library(test_loader)

    model = get_model("baseline", out_channels=4).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"Warning: {MODEL_PATH} not found. Using random weights.")

    evaluate_model(model, test_loader, ref_lib, ref_ids, args.num_phases if args.num_phases > 0 else "2-4")
