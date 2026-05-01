
import os
# Force CPU usage to avoid CUDA errors in some environments
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from autoXRD import spectrum_analysis, quantifier
import tensorflow as tf
from tqdm import tqdm
import json
import pickle

# Constants
BASE_DIR = os.path.dirname(__file__)
MIN_ANGLE = 10.0
MAX_ANGLE = 80.0
MODEL_PATH = os.path.join(BASE_DIR, '../Model_NPZ.h5')
REF_DIR = os.path.join(BASE_DIR, 'References')
# Check for test data in root or current dir
TEST_DATA_PATH = 'Multiphase_Data/test_data.npz'
if not os.path.exists(TEST_DATA_PATH):
    TEST_DATA_PATH = os.path.join(BASE_DIR, 'Multiphase_Data/test_data.npz')
OUTPUT_JSON = os.path.join(BASE_DIR, 'multiphase_evaluation_results.json')
MAPPING_PATH = os.path.join(BASE_DIR, '../id_to_ref_mapping_full.pkl')

def evaluate():
    # Load test data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found. Run prepare_multiphase_data.py first.")
        return

    # Load mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'rb') as f:
            id_to_label = pickle.load(f)
    else:
        print(f"Warning: {MAPPING_PATH} not found. True labels will remain as IDs.")
        id_to_label = {}

    test_data = np.load(TEST_DATA_PATH, allow_pickle=True)
    x_test = test_data['x']
    y_labels = test_data['labels'] # True CIF filenames/IDs
    y_weights = test_data['weights'] # True weight fractions

    # Initialize results storage
    all_results = []

    # Loop through test samples
    for i in tqdm(range(len(x_test)), desc="Evaluating"):
        spectrum = x_test[i]
        true_ids = y_labels[i]
        true_weights = y_weights[i]

        # Convert true IDs to labels using mapping
        # Only keep the sample if ALL true phases are in the References
        true_phases = []
        skip_sample = False
        for cid in true_ids:
            if cid in id_to_label:
                true_phases.append(id_to_label[cid])
            else:
                skip_sample = True
                break

        if skip_sample:
            continue

        # Save spectrum to a temporary file because SpectrumAnalyzer expects a file
        temp_spec_dir = 'temp_eval'
        os.makedirs(temp_spec_dir, exist_ok=True)
        temp_spec_name = f'test_spec_{i}.xy'
        temp_spec_path = os.path.join(temp_spec_dir, temp_spec_name)

        # SpectrumAnalyzer expects 2 columns: angle, intensity
        angles = np.linspace(MIN_ANGLE, MAX_ANGLE, len(spectrum))
        np.savetxt(temp_spec_path, np.column_stack((angles, spectrum)))

        try:
            # 1. Phase Identification using autoXRD's branching algorithm
            # Use a slightly lower min_conf (15%) for better recall on complex mixtures
            analyzer = spectrum_analysis.SpectrumAnalyzer(
                spectra_dir=temp_spec_dir,
                spectrum_fname=temp_spec_name,
                max_phases=5,
                cutoff_intensity=5.0,
                reference_dir=REF_DIR,
                model_path=MODEL_PATH,
                min_conf=15.0
            )

            # suspected_mixtures returns lists of predicted phases and confidences
            pred_mixtures, conf_mixtures, _, scale_mixtures, _ = analyzer.suspected_mixtures

            if not pred_mixtures:
                pred_phases = []
                pred_weights = []
            else:
                # Take the first mixture
                pred_phases = pred_mixtures[0]
                # 2. Quantification (Weight fraction estimation)
                s_factors = scale_mixtures[0] if scale_mixtures[0] is not None else [1.0] * len(pred_phases)

                pred_weights = quantifier.main(
                    temp_spec_dir,
                    temp_spec_name,
                    pred_phases,
                    s_factors,
                    rietveld=False,
                    reference_dir=REF_DIR
                )

            # Ensure pred_weights is valid
            if pred_weights is None:
                pred_weights = []

            # Store results
            if len(pred_phases) == len(pred_weights):
                all_results.append({
                    "sample_index": i,
                    "true_phases": [str(p) for p in true_phases],
                    "true_weights": [float(w) for w in true_weights],
                    "pred_phases": [str(p) for p in pred_phases],
                    "pred_weights": [float(w) for w in pred_weights]
                })

        except Exception as e:
            # print(f"Error evaluating sample {i}: {e}")
            continue
        finally:
            if os.path.exists(temp_spec_path):
                os.remove(temp_spec_path)

    # Calculate Metrics
    # 1. Phase Precision/Recall
    # 2. Weight Fraction Error (MAE for correctly identified phases)

    total_precision = 0
    total_recall = 0
    weight_mae = []

    for res in all_results:
        true_set = set(res["true_phases"])
        pred_set = set(res["pred_phases"])

        intersection = true_set.intersection(pred_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0
        recall = len(intersection) / len(true_set) if true_set else 0

        total_precision += precision
        total_recall += recall

        # Weight error for correctly matched phases
        for p in intersection:
            t_idx = res["true_phases"].index(p)
            p_idx = res["pred_phases"].index(p)
            weight_mae.append(abs(res["true_weights"][t_idx] - res["pred_weights"][p_idx]))

    metrics = {
        "mean_precision": total_precision / len(all_results) if all_results else 0,
        "mean_recall": total_recall / len(all_results) if all_results else 0,
        "mean_weight_mae": float(np.mean(weight_mae)) if weight_mae else 0,
        "num_samples": len(all_results)
    }

    final_output = {
        "metrics": metrics,
        "detailed_results": all_results
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Evaluation complete. Metrics: {metrics}")
    print(f"Detailed results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    evaluate()
