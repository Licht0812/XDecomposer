import json
import numpy as np
import glob
import os

metrics_files = glob.glob('test_results/rruff_kfold_results/fold_*/test_metrics.json')
if not metrics_files:
    print("No metric files found.")
    exit()

all_metrics = {}
for f in metrics_files:
    with open(f, 'r') as fp:
        m = json.load(fp)
        for k, v in m.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

print("="*50)
print(f"📊 Final Average Metrics Across {len(metrics_files)} Folds")
print("="*50)

final_agg = {}
for k, v_list in all_metrics.items():
    mean_v = np.mean(v_list)
    std_v = np.std(v_list)
    final_agg[k] = {'mean': mean_v, 'std': std_v}
    print(f"{k}: {mean_v:.4f} ± {std_v:.4f}")

with open('test_results/rruff_kfold_results/final_averaged_metrics.json', 'w') as out:
    json.dump(final_agg, out, indent=4)
print("\nSaved final aggregated results to test_results/rruff_kfold_results/final_averaged_metrics.json")
