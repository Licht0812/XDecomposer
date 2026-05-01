# XDecomposer Script Guide

Run every Bash script from the project root.
Activate `spectra` before running any launcher.

## Layout

- `bash_train/`: training launchers
- `bash_test/`: evaluation launchers
- `bash_ablation/`: ablation launchers
- `python_runners/`: Python entry points used by the Bash wrappers

## Main Scripts

- `bash_train/run_pretrain.sh`: pretrain the XRD encoder
- `bash_train/run_xdecomposer.sh`: train the FiLM-based XDecomposer model
- `bash_train/run_rruff_finetune_kfold.sh`: RRUFF 5-fold cross-validation
- `bash_test/run_mp20_k_test.sh`: test MP20 mixtures for `k=2,3,4`
- `bash_test/run_rruff_k_test_kfold.sh`: RRUFF fold-wise zero-shot evaluation
- `bash_ablation/run_ablation_yaml.sh`: train ablation variants
- `bash_ablation/run_ablation_eval.sh`: evaluate ablation checkpoints

## Examples

```bash
conda activate spectra
bash scripts/bash_train/run_pretrain.sh
bash scripts/bash_train/run_xdecomposer.sh --gpus 0,1 --name main
bash scripts/bash_train/run_rruff_finetune_kfold.sh
bash scripts/bash_test/run_mp20_k_test.sh
bash scripts/bash_test/run_rruff_k_test_kfold.sh
```

