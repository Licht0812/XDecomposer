# Transformer Baselines

This directory contains the baseline implementation requested by
`baseline_design_transformer_family.md`.

## Models

All three models use the same pretrained MAE encoder-side backbone:

```text
XRD -> patch_embed -> pos_encoding -> encoder -> latent tokens
```

Only the module after the shared backbone differs:

- `transformer`: learned source/patch queries + standard Transformer decoder with cross-attention. Its direct source head uses `softplus` output even if the shared config says `relu`, avoiding dead-ReLU collapse to all-zero spectra.
- `itransformer`: latent-space inverted Transformer, treating MAE channels as variate tokens.
- `patchtst`: encoder-only PatchTST-style source-wise prediction head.

All models output:

```text
preds: [B, K, L]
activity_logits: [B, K]
```

Patch-level predictions are reconstructed to full XRD length by overlap-add `unpatchify`.

## Loss Objective

Baseline training uses `baselines/transformer_family/losses.py` and does not
reuse `src.losses`. The objective is the effective non-zero configuration:

```text
total_loss = PIT_plain_L1 + lambda_activity * activity_BCE
```

For `baseline_name: transformer` only, the direct source decoder also uses a
code-fixed soft mixture consistency term:

```text
total_loss += 5.0 * L1(sum(pred_sources), mixture)
```

`lambda_activity` is set in `train_config.yaml`; zero-weight main-model loss
terms such as SI-SDR, geometry, and peak weighting are not part of the baseline
config. The Transformer mixture consistency weight is intentionally not exposed
in YAML because it is part of that baseline architecture choice.

## Training

Multi-GPU training uses `torch.distributed.run` and enables SwanLab by default.

Local Mac is intended for syntax/import/light forward checks only. Formal training should be
submitted on the Linux A100 cluster with Slurm. GPU and node counts are controlled by
`sbatch`, not by hardcoded `#SBATCH -N` / `#SBATCH --gres` inside the script:

```bash
sbatch -N 1 --gres=gpu:2 -p project1 -A project1 baselines/transformer_family/train.sh --baseline transformer --name transformer_mp20
```

For 1 node with 8 A100 GPUs:

```bash
sbatch -N 1 --gres=gpu:8 -p project1 -A project1 baselines/transformer_family/train.sh --baseline transformer --name transformer_mp20_8gpu
```

For 2 nodes with 8 A100 GPUs per node:

```bash
sbatch -N 2 --gres=gpu:8 -p project1 -A project1 baselines/transformer_family/train.sh --baseline transformer --name transformer_mp20_2node
```

If your account is bound to a sub-partition such as `project1-1`, override the Slurm options
at submission time:

```bash
sbatch -N 1 --gres=gpu:2 -p project1-1 -A project1 baselines/transformer_family/train.sh --baseline transformer --name transformer_mp20
```

The training script follows the A100 cluster manual:

- Loads `miniconda` through `module load miniconda` when the `module` command is available.
- Activates the `spectra` conda environment.
- Sets `PYTHONUNBUFFERED=1` for live logs.
- Sets NCCL variables recommended by the manual for A100 multi-GPU communication.

Direct run examples after entering an allocated GPU node:

```bash
baselines/transformer_family/train.sh --baseline transformer --gpus 0,1 --name transformer_mp20
baselines/transformer_family/train.sh --baseline itransformer --gpus 0,1 --name itransformer_mp20
baselines/transformer_family/train.sh --baseline patchtst --gpus 0,1 --name patchtst_mp20
```

Extra runner arguments can be appended directly, for example:

```bash
baselines/transformer_family/train.sh --baseline patchtst --gpus 0,1 --epochs 100 --batch_size 64
```

Quick smoke test on one allocated GPU node:

```bash
baselines/transformer_family/train.sh \
  --baseline patchtst \
  --gpus 0 \
  --name smoke_patchtst \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --max_train_steps 10 \
  --max_val_steps 2 \
  --vis_interval 0
```

Default training hyperparameters, SwanLab naming, and checkpoint output paths are kept in:

```text
baselines/transformer_family/train_config.yaml
```

`train.sh` is intentionally a short launcher for Slurm/DDP/environment setup.

Use a different YAML file with:

```bash
baselines/transformer_family/train.sh --config path/to/train_config.yaml --baseline patchtst --name exp_name
```

The script reads paths from `configs/paths.sh`, including:

- `PATH_DATA_SINGLEPHASE`
- `PATH_DATA_CRYSTAL_DB`
- `PATH_CKPT_PRETRAIN`
- `PATH_TEMPLATE_BASELINE_SAVE_DIR`

## Evaluation

Evaluate a trained checkpoint on pure k=2, k=3, and k=4 MP20 mixtures:

```bash
baselines/transformer_family/eval_mp20_k.sh --checkpoint checkpoints/transformer_transformer_xxx/best.pt
```

If the checkpoint config does not contain `baseline_name`, pass it explicitly:

```bash
baselines/transformer_family/eval_mp20_k.sh --checkpoint path/to/best.pt --baseline transformer
```

Evaluate the same checkpoint on RRUFF k-fold splits:

```bash
bash baselines/transformer_family/eval_rruff_kfold.sh --checkpoint path/to/best.pt --baseline transformer
```

Useful overrides:

```bash
bash baselines/transformer_family/eval_rruff_kfold.sh \
  --checkpoint path/to/best.pt \
  --baseline transformer \
  -- \
  --device cpu \
  --batch_size 128 \
  --num_folds 5 \
  --k_values 2 3 4 \
  --virtual_epoch_length 1000
```

Each MP20 run writes metrics and visualizations under the output roots defined
in `configs/paths.sh`. RRUFF
k-fold evaluation writes one `test_metrics.json` per `(k, fold)` and an
aggregate `rruff_kfold_mean_std.json` containing mean and standard deviation
for every numeric metric, including `id_acc_top1` through `id_acc_top10`.
The RRUFF evaluator builds `data/rruff_processed/*.npz` once if the cache is
missing, then all folds reuse it. Use `--rebuild_rruff_cache` after changing
RRUFF preprocessing, or `--skip_rruff_cache_build` if you want to fail fast when
the cache is missing.
