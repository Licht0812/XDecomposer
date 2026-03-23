
import os
import numpy as np
from autoXRD import cnn
import sys

# Constants
TRAIN_DATA_PATH = 'NPZ_Training_Data.npy'
MODEL_OUT_PATH = 'Model_NPZ_Subset.h5'
NUM_EPOCHS = 1
TEST_FRACTION = 0.1

def main():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Error: {TRAIN_DATA_PATH} not found.")
        return

    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    # Load only first 100 classes for testing
    full_data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
    xrd_specs = full_data[:100]
    
    print(f"Loaded {len(xrd_specs)} classes.")
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    cnn.main(xrd_specs, NUM_EPOCHS, TEST_FRACTION, is_pdf=False, fmodel=MODEL_OUT_PATH)
    
    print(f"Model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()
