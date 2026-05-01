import argparse
import glob
import json
import os

import numpy as np


METRIC_KEYS = [
    "loss",
    "si_sdr",
    "pearson_corr",
    "sir",
    "sar",
    "delta_2theta",
    "fwhm_error",
    *[f"id_acc_top{k}" for k in range(1, 11)],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="output")
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    for phase in [2, 3, 4]:
        print("===========================================")
        print(f"Aggregating results for {phase} phases")
        print("===========================================")

        metrics_dict = {key: [] for key in METRIC_KEYS}

        for fold in range(args.num_folds):
            dirs = sorted(glob.glob(f"{args.results_dir}/*_fold_{fold}"), reverse=True)
            if not dirs:
                print(f"Warning: no results found for fold {fold}")
                continue

            latest_dir = dirs[0]
            json_file = f"{latest_dir}/inference_results_{phase}_phases.json"
            if not os.path.exists(json_file):
                fallback_file = f"{latest_dir}/checkpoints/inference_results_{phase}_phases.json"
                if os.path.exists(fallback_file):
                    json_file = fallback_file

            try:
                with open(json_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)

                if not isinstance(data, dict):
                    print(f"Warning: unexpected result format in {json_file}")
                    continue

                for key in METRIC_KEYS:
                    if key in data:
                        metrics_dict[key].append(float(data[key]))
            except Exception as exc:
                print(f"Error processing {json_file}: {exc}")

        print(f"\n--- {phase} phases 5-fold average ---")
        for key in METRIC_KEYS:
            values = metrics_dict[key]
            if not values:
                continue
            print(f"{key:20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print()


if __name__ == "__main__":
    main()
