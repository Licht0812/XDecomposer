import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from datetime import datetime

from data_utils import get_dataloaders, OnlineMixingConfig
from model import get_model
from metrics_utils import SeparationLoss, calculate_all_metrics

# Configuration

config = OnlineMixingConfig(
    MIN_K=2,
    MAX_K=4,
    MIN_WEIGHT=0.15,
    XRD_LENGTH=3500,
    AUGMENT=True,
    NOISE_LEVEL=0.01,
    SEED=7
)

DB_PATH = 'data/UniqRruffCrystal.db'
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
NUM_FOLDS = 5 # 5-fold cross validation

# Training Function

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, fold=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_si_sdr = -float('inf')

    for epoch in range(num_epochs):
        print(f'Fold {fold} - Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_metrics = {
                'pearson_corr': 0.0, 'si_sdr': 0.0
            }
            count = 0
            iter_count = 0
            total_batches = len(dataloaders[phase])

            for batch in dataloaders[phase]:
                inputs = batch['multiphase_xrd'].to(DEVICE)   # [B, 1, L]
                targets = batch['single_xrds'].to(DEVICE)     # [B, K, L]
                phase_ids = batch['phase_ids'].to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # [B, K, L]
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Calculate metrics for the batch (Training/Val only use waveform metrics)
                batch_metrics = calculate_all_metrics(
                    outputs.detach(), targets, phase_ids
                )
                for k in running_metrics:
                    if k in batch_metrics:
                        running_metrics[k] += batch_metrics[k] * inputs.size(0)

                count += inputs.size(0)
                iter_count += 1

                if iter_count % 100 == 0:
                    current_loss = running_loss / count
                    print(f'  [{phase}] Batch {iter_count}/{total_batches} | Loss: {current_loss:.4f} SI-SDR: {batch_metrics.get("si_sdr", 0):.4f}')

            epoch_loss = running_loss / count
            epoch_metrics = {k: v / count for k, v in running_metrics.items()}

            print(
                f'{phase} Loss: {epoch_loss:.4f} '
                f'SI-SDR: {epoch_metrics["si_sdr"]:.4f} '
                f'Pearson: {epoch_metrics["pearson_corr"]:.4f}'
            )

            # Deep copy the model if it's the best (based on SI-SDR)
            if phase == 'val' and epoch_metrics['si_sdr'] > best_si_sdr:
                best_si_sdr = epoch_metrics['si_sdr']
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'best_separation_model_fold_{fold}.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Fold {fold} Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Fold {fold} Best val SI-SDR: {best_si_sdr:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Main

if __name__ == "__main__":
    for fold in range(NUM_FOLDS):
        print(f"\\n{'='*40}")
        print(f"Starting Training for Fold {fold}")
        print(f"{'='*40}\\n")

        # 1. Create Dataloaders for this fold
        train_loader, val_loader, test_loader = get_dataloaders(
            DB_PATH, config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            num_folds=NUM_FOLDS, fold=fold
        )

        dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

        # 2. Create Model (BaselineSeparationNet: Input [B, 1, L] -> Output [B, 4, L])
        model = get_model("baseline", out_channels=config.MAX_K).to(DEVICE)

        # 3. Define Loss and Optimizer
        criterion = SeparationLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 4. Train
        train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS, fold=fold)
