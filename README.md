
# XDecomposer



<p align="center">
  <b>Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction</b>
</p>

<p align="center">
  <a href="https://github.com/Licht0812/XRDUnmix/stargazers">
    <img src="https://img.shields.io/github/stars/Licht0812/XRDUnmix?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Licht0812/XRDUnmix/network/members">
    <img src="https://img.shields.io/github/forks/Licht0812/XRDUnmix?style=social" alt="GitHub Forks">
  </a>
  <a href="https://github.com/Licht0812/XRDUnmix/issues">
    <img src="https://img.shields.io/github/issues/Licht0812/XRDUnmix" alt="Issues">
  </a>
  <a href="https://github.com/Licht0812/XRDUnmix/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Licht0812/XRDUnmix" alt="License">
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

Pretrained checkpoints and the RRUFF dataset are available via Google Drive.

Please download and extract them into your local **XDecomposer/** directory.

**Google Drive Download** [Link](https://drive.google.com/drive/folders/1bdgzdtouRn7TObqmvMgNLdA5PGPx0dVt?usp=sharing)


---

## Installation

```bash
git clone https://github.com/Licht0812/XRDUnmix.git
cd XRDUnmix
conda create -n xdecomposer python=3.10
conda activate xdecomposer
pip install -r requirements.txt
````

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

```bash
bash scripts/bash_train/run_pretrain.sh
```

Resume:

```bash
bash scripts/bash_train/run_pretrain.sh /path/to/checkpoint
```

### Separation Backbone Training

```bash
bash scripts/bash_train/run_separation.sh --gpus 0,1 --name my_experiment
```

### RRUFF Fine-tuning (5-Fold CV)

```bash
bash scripts/bash_train/run_rruff_kfold.sh
```

---

## 2. Evaluation

### Default Evaluation

```bash
bash scripts/bash_test/run_default_eval.sh
```

### MP20 Multiphase Benchmark

```bash
bash scripts/bash_test/run_mp20_k_test.sh
```

### RRUFF Real-world Evaluation

```bash
bash scripts/bash_test/run_rruff_k_test.sh
```

---

## Repository Structure

```text
XDecomposer/
├── configs/                 # Path / environment settings
│   └── paths.sh
│
├── scripts/
│   ├── bash_train/          # Training scripts
│   ├── bash_test/           # Evaluation scripts
│   ├── bash_search/         # Hyperparameter search
│   ├── bash_ablation/       # Ablation studies
│   ├── python_runners/      # Python entrypoints
│   └── tools/              # Utilities
│
├── src/                    # Core model source code
├── checkpoints/            # Saved weights
├── data/                   # Local datasets
└── docs/                   # Figures / docs
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

