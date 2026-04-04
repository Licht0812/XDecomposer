import os
# Force CPU usage as GPU environment is unstable (PTX version errors)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pickle
import random
from autoXRD import cnn
from tqdm import tqdm
from scipy.interpolate import interp1d

# =============================================================================
# Configuration
# =============================================================================

NPZ_DIR = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data'
MAPPING_PATH = 'id_to_ref_mapping_full.pkl'
MODEL_OUT_PATH = 'Model_NPZ_Full.h5'
BATCH_SIZE = 128
NUM_EPOCHS = 150
TARGET_LENGTH = 3500
SEED = 7

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class XRDDataGenerator(Sequence):
    """
    Robust Data Generator for Large-Scale XRD Classification.
    Features:
    - Pre-scans for valid files.
    - Randomly selects 1 of 20 samples per CID each time it's accessed (Data Augmentation).
    - Interpolates to target length.
    """
    def __init__(self, id_to_ref, ref_list, batch_size=128, shuffle=True):
        self.ref_to_index = {ref: i for i, ref in enumerate(ref_list)}
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Pre-scan: Group available samples by CID
        self.cid_to_samples = {}
        print("Pre-scanning NPZ directory for valid samples...")
        
        # We only consider CIDs present in our mapping
        target_cids = set(id_to_ref.keys())
        
        # Fast scan of directory
        all_files = os.listdir(NPZ_DIR)
        for fname in tqdm(all_files):
            if fname.startswith('crystal_') and fname.endswith('.npz'):
                # Extract row_id and Sample Index
                # Format: crystal_12345_sample_01.npz
                parts = fname.split('_')
                try:
                    row_id = int(parts[1])
                    # Label = row_id - 1
                    label = row_id - 1
                    if label in target_cids:
                        s_idx = int(parts[3].split('.')[0])
                        if label not in self.cid_to_samples:
                            self.cid_to_samples[label] = []
                        self.cid_to_samples[label].append(s_idx)
                except (ValueError, IndexError):
                    continue
        
        # Filter mapping to only include CIDs that actually have files
        self.valid_cids = sorted(list(self.cid_to_samples.keys()))
        self.labels = [self.ref_to_index[id_to_ref[cid]] for cid in self.valid_cids]
        
        print(f"Found valid data for {len(self.valid_cids)} / {len(target_cids)} crystal IDs.")
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.valid_cids) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_cids = [self.valid_cids[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        
        X, y = self.__data_generation(batch_cids, batch_labels)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.valid_cids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_cids, batch_labels):
        X = np.empty((self.batch_size, TARGET_LENGTH, 1), dtype=np.float32)
        y = np.array(batch_labels, dtype=np.int32)

        for i, label in enumerate(batch_cids):
            # Label is row.id - 1, so row.id = label + 1
            row_id = label + 1
            # Randomly pick ONE of the available samples for this Label
            s_idx = random.choice(self.cid_to_samples[label])
            fpath = os.path.join(NPZ_DIR, f"crystal_{row_id}_sample_{s_idx:02d}.npz")
            
            try:
                data = np.load(fpath)
                sig = data['y'] if 'y' in data else data['intensity']
                
                # Length check and interpolation
                if len(sig) != TARGET_LENGTH:
                    x_old = np.linspace(10.0, 80.0, len(sig))
                    x_new = np.linspace(10.0, 80.0, TARGET_LENGTH)
                    sig = np.interp(x_new, x_old, sig)
                
                # Normalization (Match autoXRD's 0-100 scaling)
                if np.max(sig) > 0:
                    sig = 100.0 * (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
                
                X[i, :, 0] = sig
            except Exception:
                # Fallback for corrupted files
                X[i, :, 0] = np.zeros(TARGET_LENGTH)

        return X, y

def main():
    if not os.path.exists(MAPPING_PATH):
        print(f"Error: {MAPPING_PATH} not found. Please run rebuild_dataset.py first.")
        return

    with open(MAPPING_PATH, 'rb') as f:
        id_to_ref = pickle.load(f)
    
    # Reference list (sorted to define class indices)
    REF_DIR = 'Novel-Space/References'
    ref_list = sorted([f for f in os.listdir(REF_DIR) if f.endswith('.cif')])
    num_classes = len(ref_list)
    print(f"Total target classes: {num_classes}")
    
    # Save the class list for reference during evaluation
    with open('class_list.pkl', 'wb') as f:
        pickle.dump(ref_list, f)

    # Per user request: ALL crystal IDs participate in training.
    all_cids = sorted(list(id_to_ref.keys()))
    
    # Use Seed 7 for reproducibility in split
    random.seed(SEED)
    random.shuffle(all_cids)
    
    # train_cids_dict = {cid: id_to_ref[cid] for cid in all_cids}
    
    # For validation, we pick a small random subset of all_cids (1%)
    split_idx = int(0.99 * len(all_cids))
    train_cids_dict = {cid: id_to_ref[cid] for cid in all_cids[:split_idx]}
    val_cids_dict = {cid: id_to_ref[cid] for cid in all_cids[split_idx:]}

    train_gen = XRDDataGenerator(train_cids_dict, ref_list, batch_size=BATCH_SIZE)
    val_gen = XRDDataGenerator(val_cids_dict, ref_list, batch_size=BATCH_SIZE)
  
    print(f"Building model for {num_classes} classes...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Input
    
    model = Sequential([
        Input(shape=(TARGET_LENGTH, 1)),
        Conv1D(31, 100, strides=5, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(11, 50, strides=5, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2, padding='same'),
        Conv1D(7, 25, strides=5, padding='same', activation='relu'),
        MaxPooling1D(3, strides=2, padding='same'),
        Flatten(),
        Dense(3100, activation='relu'),
        BatchNormalization(),
        cnn.CustomDropout(0.7),
        Dense(1200, activation='relu'),
        BatchNormalization(),
        cnn.CustomDropout(0.7),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae']
    )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print(f"Starting training. Results will be saved to {MODEL_OUT_PATH}")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )

    print(f"Training complete. Best model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
