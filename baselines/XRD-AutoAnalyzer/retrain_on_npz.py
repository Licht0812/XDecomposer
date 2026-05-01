
import os
import numpy as np
from autoXRD import cnn
import sys

# Constants
TRAIN_DATA_PATH = 'NPZ_Training_Data.npy'
MODEL_OUT_PATH = 'Model_NPZ.h5'
NUM_EPOCHS = 50
TEST_FRACTION = 0.1

def main():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: {TRAIN_DATA_PATH} not found. Run prepare_npz_training_data.py first.")
        return

    print(f"Loading training data from {TRAIN_DATA_PATH}...", flush=True)
    # data is a list of lists: N_classes x M_aug x 4501
    xrd_specs = np.load(TRAIN_DATA_PATH, allow_pickle=True)

    print(f"Loaded {len(xrd_specs)} classes.", flush=True)

    # Train the model
    # Note: autoXRD.cnn.main handles splitting and training
    print(f"Starting training for {NUM_EPOCHS} epochs...", flush=True)
    try:
        cnn.main(xrd_specs, NUM_EPOCHS, TEST_FRACTION, is_pdf=False, fmodel=MODEL_OUT_PATH)
        print(f"Model saved to {MODEL_OUT_PATH}", flush=True)
    except Exception as e:
        print(f"Training failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Force CPU if needed or set GPU growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()
