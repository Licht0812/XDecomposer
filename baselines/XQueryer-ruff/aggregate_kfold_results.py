import json
import glob
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='output')
    parser.add_argument('--num_folds', type=int, default=5)
    args = parser.parse_args()

    for phase in [2, 3, 4]:
        print(f"===========================================")
        print(f"📊 Aggregating Results for {phase} phases")
        print(f"===========================================")
        
        metrics_dict = {
            'top1_acc': [],
            'top5_acc': [],
            'top10_acc': [],
            'rwp': [],
            'pearson': [],
            'sisdr': [],
            'sir': [],
            'sar': [],
            'delta_2theta': [],
            'fwhm_error': [],
            'intensity_consistency': [],
            'quant_mae': [],
            'act_f1': [],
            'act_precision': [],
            'act_recall': [],
            'act_exact_match': [],
            'oracle_exact_match': []
        }
        
        for fold in range(args.num_folds):
            # Find the latest output directory for this fold
            dirs = sorted(glob.glob(f"{args.results_dir}/*_fold_{fold}"), reverse=True)
            if not dirs:
                print(f"Warning: No results found for fold {fold}")
                continue
            
            latest_dir = dirs[0]
            json_file = f"{latest_dir}/inference_results_{phase}_phases.json"
            
            # Check fallback path if not found (in case it was saved inside checkpoints/)
            if not os.path.exists(json_file):
                fallback_file = f"{latest_dir}/checkpoints/inference_results_{phase}_phases.json"
                if os.path.exists(fallback_file):
                    json_file = fallback_file
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # We need to recalculate averages from the JSON since the JSON stores sample-level data
                total_samples = len(data)
                total_phases = sum(len(sample['phases']) for sample in data)
                
                if total_samples == 0 or total_phases == 0:
                    print(f"Warning: Empty results for fold {fold}")
                    continue
                
                # We can either recalculate or parse the stdout of infer.py.
                # Since we have JSON, let's recalculate the averages.
                top1_hits, top5_hits, top10_hits = 0, 0, 0
                rwp_sum, pearson_sum, sisdr_sum = 0, 0, 0
                sir_sum, sar_sum, delta_2theta_sum = 0, 0, 0
                fwhm_error_sum, intensity_consistency_sum, quant_mae_sum = 0, 0, 0
                
                for sample in data:
                    for p in sample['phases']:
                        target_id = p['gt_id']
                        top10_preds = [pred['id'] for pred in p['top10_preds']]
                        
                        if len(top10_preds) > 0 and target_id == top10_preds[0]: top1_hits += 1
                        if target_id in top10_preds[:5]: top5_hits += 1
                        if target_id in top10_preds[:10]: top10_hits += 1
                        
                        rwp_sum += p['rwp']
                        pearson_sum += p['pearson']
                        sisdr_sum += p['sisdr']
                        sir_sum += p['sir']
                        sar_sum += p['sar']
                        delta_2theta_sum += p['delta_2theta']
                        fwhm_error_sum += p['fwhm_error']
                        intensity_consistency_sum += p['intensity_consistency']
                        # Quant MAE uses the pct derived from pattern integrals
                        quant_mae_sum += abs(p.get('pred_pct', p['pred_ratio']) - p.get('target_pct', p['gt_ratio'])) * 100

                metrics_dict['top1_acc'].append(top1_hits / total_phases)
                metrics_dict['top5_acc'].append(top5_hits / total_phases)
                metrics_dict['top10_acc'].append(top10_hits / total_phases)
                metrics_dict['rwp'].append(rwp_sum / total_phases)
                metrics_dict['pearson'].append(pearson_sum / total_phases)
                metrics_dict['sisdr'].append(sisdr_sum / total_phases)
                metrics_dict['sir'].append(sir_sum / total_phases)
                metrics_dict['sar'].append(sar_sum / total_phases)
                metrics_dict['delta_2theta'].append(delta_2theta_sum / total_phases)
                metrics_dict['fwhm_error'].append(fwhm_error_sum / total_phases)
                metrics_dict['intensity_consistency'].append(intensity_consistency_sum / total_phases)
                metrics_dict['quant_mae'].append(quant_mae_sum / total_phases)
                
                # Activity metrics are sample level, which are not directly saved in JSON
                # We could modify infer.py to save those, but for now we focus on the available ones.
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        # Compute averages
        print(f"\n--- {phase} Phases 5-Fold Average ---")
        for k, v in metrics_dict.items():
            if len(v) > 0:
                mean_val = np.mean(v)
                std_val = np.std(v)
                if k == 'quant_mae':
                    print(f"{k:25}: {mean_val:.2f}% ± {std_val:.2f}%")
                elif 'acc' in k:
                    print(f"{k:25}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"{k:25}: {mean_val:.4f} ± {std_val:.4f}")
        print("\n")

if __name__ == "__main__":
    main()
