# XRDUnmix Scripts Directory Guide

All experimental and execution scripts are now organized into specific subdirectories. **Please ensure all Bash scripts are executed from the project root directory `XRDUnmix/`**, rather than from within the subdirectories, to avoid path errors.

## Directory Structure Overview

* **`bash_train/`**: Contains scripts for official model training executions.
  * `run_pretrain.sh`: Configures and launches MAE pre-training tasks.
  * `run_separation.sh`: Configures and launches baseline separation training tasks.
  * `run_rruff_kfold.sh`: Configures and launches 5-fold cross-validation fine-tuning on the RRUFF database (with automatic metrics aggregation).
* **`bash_test/`**: Contains scripts for evaluation and metric generation.
  * `run_default_eval.sh`: Executes basic default metric testing.
  * `run_mp20_k_test.sh`: Evaluates various multiphase mixtures using the MP20 test set.
  * `run_rruff_k_test.sh`: Inference and evaluation of various multiphase mixtures on the real-world RRUFF test set.
* **`bash_ablation/`**: Contains ablation study scripts (e.g., comparing different loss combinations, module removals).
* **`bash_search/`**: Contains hyperparameter search scripts.
  * `run_grid_search.sh`: Grid search evaluation for mathematical thresholds (e.g., alpha, margin, hard_threshold).
* **`python_runners/`**: **[DO NOT RUN FILES IN THIS DIRECTORY DIRECTLY]**
  * Stores the actual Python execution entry points (e.g., `train_pretrain.py`, `test_separation_film.py`, `finetune_rruff.py`, inference script `find_50_perfect.py`, etc.). These are solely invoked by the outer `bash_*/` scripts.
* **`tools/`**: A collection of utility tools for diagnostics, data processing, and CIF file validation.

## Execution Examples

Always launch scripts via bash from the `XRDUnmix/` root directory. Below are the usage commands for every listed script:

### Training Scripts
```bash
# Launch MAE pre-training (Extracting single-phase priors)
bash scripts/bash_train/run_pretrain.sh

# Launch baseline separation training (requires GPU selection and experiment name)
bash scripts/bash_train/run_separation.sh --gpus 0,1 --name "my_experiment"

# Launch RRUFF fine-tuning (5-Fold cross-validation)
bash scripts/bash_train/run_rruff_kfold.sh
```

### Testing & Evaluation Scripts
```bash
# Launch default baseline evaluation
bash scripts/bash_test/run_default_eval.sh

# Launch multiphase tests on simulated MP20 test set
bash scripts/bash_test/run_mp20_k_test.sh

# Launch multiphase tests on real-world RRUFF test set
bash scripts/bash_test/run_rruff_k_test.sh
```

### Search & Auxiliary Scripts
```bash
# Launch grid search for optimal evaluation thresholds
bash scripts/bash_search/run_grid_search.sh
```

All hyperparameter configurations and dependent weight paths are embedded into their corresponding `.sh` files via environment variables or parameter configurations. If you need to change the database source (e.g., ASE db) or specify computing devices (e.g., single GPU after salloc), simply edit the corresponding `.sh` file or the global `configs/paths.sh`.
