<p align="center">
  <b>XDecomposer</b>: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction
</p>

<p align="center">
  <a href="https://github.com/your-org/XDecomposer/stargazers">
    <img src="https://img.shields.io/github/stars/your-org/XDecomposer?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/your-org/XDecomposer/network/members">
    <img src="https://img.shields.io/github/forks/your-org/XDecomposer?style=social" alt="GitHub Forks">
  </a>
  <a href="https://github.com/your-org/XDecomposer/issues">
    <img src="https://img.shields.io/github/issues/your-org/XDecomposer" alt="Issues">
  </a>
  <a href="https://github.com/your-org/XDecomposer/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/your-org/XDecomposer" alt="License">
  </a>
</p>

---

## Project Overview

**XDecomposer** is a novel deep learning framework for **multiphase whole-pattern decomposition** in X-ray diffraction (XRD).

It is designed to separate complex mixed diffraction signals into interpretable individual phase patterns, enabling automated mineralogical and crystallographic analysis.

The framework integrates:

- Self-supervised representation learning
- Supervised decomposition modeling
- Fine-tuning on real experimental data
- Unified training / evaluation pipelines

---

## Pre-trained Models & Dataset

Pretrained checkpoints and the RRUFF dataset are available through an anonymized OSF view-only link.

Please download and extract them into your local **XDecomposer/** directory.

**OSF Download:** [Link](https://osf.io/zkunc/overview?view_only=cfd48c9fcec24983905f134bbf560392)

---

## Installation

```bash
git clone https://github.com/your-org/XDecomposer.git
cd XDecomposer
conda create -n xdecomposer python=3.10
conda activate xdecomposer
pip install -r requirements.txt
```

---

## Configuration

All environment variables and paths are centrally managed in:

```bash
configs/paths.sh
```

This acts as the **single source of truth** for the project.

### Main Settings

| Category          | Variables                                  |
| ----------------- | ------------------------------------------ |
| Conda Environment | `CONDA_ACTIVATE_PATH`, `CONDA_ENV_NAME`    |
| Dataset Paths     | `PATH_DATA_SINGLEPHASE`, `PATH_DATA_RRUFF` |
| Checkpoints       | `PATH_CKPT_MAE`, `PATH_CKPT_SEP`           |
| Output Folder     | `PATH_OUTPUT_TEST`                         |

---

## Usage Guide

## 1. Training

### MAE Pretraining

The encoder pretraining code is in `scripts/python_runners/train_pretrain.py` and `src/models/xrd_transformer.py`.

```bash
bash scripts/bash_train/run_pretrain.sh
```

### Separation Backbone Training

The training runner is `scripts/python_runners/train_xdecomposer.py`.

```bash
bash scripts/bash_train/run_xdecomposer.sh --gpus 0,1 --name main
```

### RRUFF Fine-tuning (5-Fold CV)

```bash
bash scripts/bash_train/run_rruff_finetune_kfold.sh
```

---

## 2. Evaluation

### MP20 Multiphase Benchmark

```bash
bash scripts/bash_test/run_mp20_k_test.sh
```

### RRUFF Real-world Evaluation

```bash
bash scripts/bash_test/run_rruff_k_test_kfold.sh
```

---

## Repository Structure

```text
XDecomposer/
├── configs/                 # Path / environment settings
│   ├── ablation_configs/
│   └── paths.sh
│
├── scripts/
│   ├── bash_train/          # Training scripts
│   ├── bash_test/           # Evaluation scripts
│   ├── bash_ablation/       # Ablation studies
│   └── python_runners/      # Python entrypoints
│
├── src/                    # Core model source code
├── tutorial/               # Tutorial examples
└── checkpoints/            # Saved weights
```

---

## Key Features

* End-to-end multiphase XRD decomposition
* Self-supervised feature pretraining
* Real-world dataset adaptation
* Modular research-friendly design
* Easy bash-based training pipelines
* Benchmark-ready evaluation tools

---

## Citation

If you use **XDecomposer** in your research, please cite:

```bibtex

```
