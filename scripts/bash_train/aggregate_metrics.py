import json
import numpy as np
import glob
import os
import sys

if len(sys.argv) > 1:
    base_dir = sys.argv[1]
else:
    base_dir = os.environ.get("PATH_OUTPUT_RRUFF_KFOLD_ROOT", "test_results/rruff_kfold")

metrics_pattern = os.path.join(base_dir, 'fold_*/test_metrics.json')
metrics_files = glob.glob(metrics_pattern)

if not metrics_files:
    print(f"No metric files found in {metrics_pattern}.")
    sys.exit(1)

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

out_path = os.path.join(base_dir, 'final_averaged_metrics.json')
with open(out_path, 'w') as out:
    json.dump(final_agg, out, indent=4)
print(f"\nSaved final aggregated results to {out_path}")
