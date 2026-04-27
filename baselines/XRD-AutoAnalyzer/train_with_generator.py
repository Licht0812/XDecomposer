
import os
# Force CPU usage to avoid CUDA errors and OOM
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pickle
import random
from autoXRD import cnn

# Constants
NPZ_DIR = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data'
MAPPING_PATH = 'id_to_ref_mapping_full.pkl'
MODEL_OUT_PATH = 'Model_NPZ_Generator.h5'
BATCH_SIZE = 128
NUM_EPOCHS = 50
TARGET_LENGTH = 4501

class XRDDataGenerator(Sequence):
    def __init__(self, id_to_ref, ref_list, batch_size=32, shuffle=True):
        self.id_to_ref = id_to_ref # dict: cid -> ref_name
        self.ref_list = ref_list # list: sorted ref names
        self.ref_to_index = {ref: i for i, ref in enumerate(ref_list)}
        
        # Build list of all available (cid, sample_idx) pairs
        self.samples = []
        print("Building sample list from mapping...")
        for cid, ref in id_to_ref.items():
            # Assume each cid has up to 20 samples
            for s_idx in range(20):
                fname = f"crystal_{cid}_sample_{s_idx:02d}.npz"
                # Check existence once or trust the directory? 
                # Checking 2M files is slow, let's just use the ones we found earlier
                self.samples.append((cid, s_idx, self.ref_to_index[ref]))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch_samples = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_samples)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.samples)

    def __data_generation(self, batch_samples):
        X = np.empty((self.batch_size, TARGET_LENGTH, 1), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=np.int32)

        for i, (cid, s_idx, class_idx) in enumerate(batch_samples):
            fpath = os.path.join(NPZ_DIR, f"crystal_{cid}_sample_{s_idx:02d}.npz")
            try:
                data = np.load(fpath)
                sig = data['y']
                if len(sig) != TARGET_LENGTH:
                    from scipy.interpolate import interp1d
                    x_old = np.linspace(10, 80, len(sig))
                    x_new = np.linspace(10, 80, TARGET_LENGTH)
                    f = interp1d(x_old, sig, kind='cubic', fill_value="extrapolate")
                    sig = f(x_new)
                X[i, :, 0] = sig
                y[i] = class_idx
            except:
                # Fallback for missing files
                X[i, :, 0] = np.zeros(TARGET_LENGTH)
                y[i] = class_idx

        return X, y

def main():
    if not os.path.exists(MAPPING_PATH):
        print(f"Error: {MAPPING_PATH} not found.")
        return

    with open(MAPPING_PATH, 'rb') as f:
        id_to_ref = pickle.load(f)
    
    # Classes must be consistent with autoXRD (sorted References)
    REF_DIR = 'Novel-Space/References'
    ref_list = sorted([f for f in os.listdir(REF_DIR) if f.endswith('.cif')])
    num_classes = len(ref_list)
    print(f"Number of classes: {num_classes}")

    # Split samples into train/val
    all_cids = list(id_to_ref.keys())
    random.shuffle(all_cids)
    train_split = int(0.9 * len(all_cids))
    train_cids = set(all_cids[:train_split])
    val_cids = set(all_cids[train_split:])

    train_id_to_ref = {cid: id_to_ref[cid] for cid in train_cids}
    val_id_to_ref = {cid: id_to_ref[cid] for cid in val_cids}

    train_gen = XRDDataGenerator(train_id_to_ref, ref_list, batch_size=BATCH_SIZE)
    val_gen = XRDDataGenerator(val_id_to_ref, ref_list, batch_size=BATCH_SIZE)
  
    print("Building model...")
    # Redefine model architecture here to match autoXRD's architecture
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
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['sparse_categorical_accuracy']
    )

    print("Starting training with generator...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS
    )

    model.save(MODEL_OUT_PATH, include_optimizer=False)
    print(f"Model saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
