# XRDUnmix

## 📖 Project Overview
XRDUnmix is a deep learning-based framework for the separation and phase identification of multiphase X-ray Diffraction (XRD) patterns. 

## 📂 Project Structure
To maintain a clean engineering structure, all execution-level logic has been decoupled. The overall project architecture is as follows:
- `configs/`: Contains the global environment and path configuration files for the project. The core is `paths.sh`.
- `scripts/`: Contains all command execution scripts, finely categorized into:
  - `bash_train/`: Deep learning model training scripts (including MAE pre-training, main separation model training, and fine-tuning).
  - `bash_test/`: Benchmark testing and real-world validation scripts for various evaluation scenarios.
  - `bash_search/`: Hyperparameter grid search scripts.
  - `bash_ablation/`: Ablation study scripts for verifying model performance.
  - `python_runners/`: The actual Python entry points called by the underlying Bash scripts (e.g., `train_separation_film.py` and `test_separation_film.py`).
  - `tools/`: Subsidiary tools for validation analysis, file comparison, etc. (e.g., `compare_cif_model.py`).
- `src/`: Core deep learning source code (network model definitions, network architecture, data loaders, and general utility functions).
- `checkpoints/`: Stores archived model weights during the training process (if generated).
- `data/`: Stores locally dependent database files (e.g., `rruff.db`).

## 💾 Pre-trained Models & Data
The model weights and the RRUFF dataset are available on Google Drive. Please download and extract them into your local `XRDUnmix` directory:
- [Click here to download from Google Drive](https://drive.google.com/drive/folders/1bdgzdtouRn7TObqmvMgNLdA5PGPx0dVt?usp=sharing)

## ⚙️ Configuration (Environment & Paths)
The project uses a "single source of truth" configuration managed entirely in **`configs/paths.sh`**. You can say goodbye to scattered hard-coded variables and maintain all paths in one place.
Before using any script, please check and edit `configs/paths.sh` according to your server environment:
1. **Environment Configuration (`CONDA_ACTIVATE_PATH`, `CONDA_ENV_NAME`)**: Points to the Anaconda/Miniconda `activate` path on your server and the virtual environment you want to invoke. It also automatically fixes specific CUDA dynamic library errors via `$LD_LIBRARY_PATH`.
2. **Dataset Paths (`PATH_DATA_SINGLEPHASE`, `PATH_DATA_RRUFF`, etc.)**: Points to the MP20 dataset or RRUFF crystal-derived database files. Both absolute and relative paths are supported.
3. **Models & Weights (`PATH_CKPT_MAE`, `PATH_CKPT_SEP`)**: Downstream testing scripts will automatically link to the best weight files here. Modifying this single configuration will globally change the model used for testing!
4. **Output Directory (`PATH_OUTPUT_TEST`)**: Centrally controls the starting export directory for evaluation logs, visualizations, and generated charts (default is `test_results/`).

## 🚀 Usage Guide

### 1. Training
The scripts come with readable input parameter parsing and log export support.

- **MAE Pre-training**
  Enhances the network's capability to extract single-phase XRD features via unsupervised learning.
  ```bash
  bash scripts/bash_train/run_pretrain.sh
  ```
  *(Note: For resuming training, you can pass the resume parameter, e.g., `bash scripts/bash_train/run_pretrain.sh /path/to/checkpoint`)*

- **Separation Backbone Training**
  Uses the prior knowledge extracted by MAE as a reference to train the Unmix separation component targeted at real mixture patterns.
  ```bash
  bash scripts/bash_train/run_separation.sh --gpus 0,1 --name my_experiment
  ```

- **RRUFF Fine-tuning (5-Fold Cross Validation)**
  Trains a fine-tuned model specifically for real-world collected datasets via K-fold cross-validation and includes comprehensive statistical testing:
  ```bash
  bash scripts/bash_train/run_rruff_kfold.sh
  ```

### 2. Evaluation & Testing
Because environment variables are unified, all testing scripts can be launched with ease. The scripts below will all utilize the `PATH_CKPT_SEP` specified in your `configs/paths.sh`.

- **Default Evaluation Workflow** (Comprehensive test configured with historical standard parameters/conditions):
  ```bash
  bash scripts/bash_test/run_default_eval.sh
  ```
- **Multiphase Mixture Evaluation for the MP20 Dataset** (Systematically evaluates complex scenarios of pure 2-phase, 3-phase, and 4-phase mixtures):
  ```bash
  bash scripts/bash_test/run_mp20_k_test.sh
  ```
- **RRUFF Real-World Data Evaluation**
  Inferences and evaluates multiphase mixtures using the derived RRUFF target test datasets:
  ```bash
  bash scripts/bash_test/run_rruff_k_test.sh
  ```
---
