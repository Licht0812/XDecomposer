#!/bin/bash
source "$(dirname "$0")/../../configs/paths.sh"
source $CONDA_ACTIVATE_PATH $CONDA_ENV_NAME
cd "$(dirname "$0")/../.."

alphas=(0.5 0.6 0.7 0.8)
margins=(2 3 5)
thresholds=(0.5 0.6)

if [ ! -f grid_search_results.csv ]; then
    echo "alpha, margin, threshold, id_acc_top1, id_acc_top3, id_acc_top5, id_acc_top10" > grid_search_results.csv
fi

for a in "${alphas[@]}"; do
    for m in "${margins[@]}"; do
        for t in "${thresholds[@]}"; do
            if grep -q "^$a, $m, $t," grid_search_results.csv; then
                echo "Skipping alpha=$a, margin=$m, hard_threshold=$t (already completed)"
                continue
            fi
            
            echo "Running test with alpha=$a, margin=$m, hard_threshold=$t"
            save_dir="test_results_film_val_a${a}_m${m}_t${t}"
            
            python scripts/python_runners/test_separation_film.py \
                --checkpoint $PATH_CKPT_SEP \
                --data_dir $PATH_DATA_SINGLEPHASE \
                --crystal_db $PATH_DATA_CRYSTAL_DB \
                --save_dir $save_dir \
                --alpha $a \
                --margin $m \
                --hard_threshold $t \
                --split val \
                --min_k 2 \
                --max_k 4 \
                --k_weights 0.6 0.25 0.15 > $save_dir.log 2>&1
                
            metrics_file="${save_dir}/test_metrics.json"
            if [ -f "$metrics_file" ]; then
                id_acc1=$(grep '"id_acc_top1"' "$metrics_file" | awk -F': ' '{print $2}' | tr -d ',')
                id_acc3=$(grep '"id_acc_top3"' "$metrics_file" | awk -F': ' '{print $2}' | tr -d ',')
                id_acc5=$(grep '"id_acc_top5"' "$metrics_file" | awk -F': ' '{print $2}' | tr -d ',')
                id_acc10=$(grep '"id_acc_top10"' "$metrics_file" | awk -F': ' '{print $2}' | tr -d ',')
                
                echo "$a, $m, $t, $id_acc1, $id_acc3, $id_acc5, $id_acc10" >> grid_search_results.csv
                echo "Result: Top1=${id_acc1}, Top3=${id_acc3}, Top10=${id_acc10}"
            else
                echo "Failed to get metrics!"
            fi
        done
    done
done
