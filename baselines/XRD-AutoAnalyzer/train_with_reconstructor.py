import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import pickle
import random
from tqdm import tqdm

NPZ_DIR = '/data/group/project1/Crystal/UniqCry/mp20-xrd_data/data'
MAPPING_PATH = 'id_to_ref_mapping_full.pkl'
MODEL_OUT_PATH = 'Model_Reconstructor.h5'
BATCH_SIZE = 128
NUM_EPOCHS = 100
TARGET_LENGTH = 3500
SEED = 7

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class XRDReconstructionGenerator(Sequence):
    def __init__(self, id_to_ref, ref_list, batch_size=128, shuffle=True):
        self.ref_to_index = {ref: i for i, ref in enumerate(ref_list)}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cid_to_samples = {}
        target_cids = set(id_to_ref.keys())
        all_files = os.listdir(NPZ_DIR)
        for fname in tqdm(all_files, desc="Scanning NPZ"):
            if fname.startswith('crystal_') and fname.endswith('.npz'):
                parts = fname.split('_')
                try:
                    row_id = int(parts[1])
                    label = row_id - 1
                    if label in target_cids:
                        s_idx = int(parts[3].split('.')[0])
                        if label not in self.cid_to_samples: self.cid_to_samples[label] = []
                        self.cid_to_samples[label].append(s_idx)
                except: continue
        self.valid_cids = sorted(list(self.cid_to_samples.keys()))
        self.on_epoch_end()
    def __len__(self): return int(np.floor(len(self.valid_cids) / self.batch_size))
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_cids = [self.valid_cids[i] for i in indices]
        return self.__data_generation(batch_cids)
    def on_epoch_end(self):
        self.indices = np.arange(len(self.valid_cids))
        if self.shuffle: np.random.shuffle(self.indices)
    def __data_generation(self, batch_cids):
        X = np.empty((self.batch_size, TARGET_LENGTH, 1), dtype=np.float32)
        y = np.empty((self.batch_size, TARGET_LENGTH), dtype=np.float32)
        for i, label in enumerate(batch_cids):
            row_id = label + 1
            s_idx = random.choice(self.cid_to_samples[label])
            for sig_type, s_idx_val, arr, is_x in [('noisy', s_idx, X, True), ('clean', 0, y, False)]:
                fpath = os.path.join(NPZ_DIR, f"crystal_{row_id}_sample_{s_idx_val:02d}.npz")
                try:
                    data = np.load(fpath)
                    sig = data['y'] if 'y' in data else data['intensity']
                    if len(sig) != TARGET_LENGTH:
                        sig = np.interp(np.linspace(10, 80, TARGET_LENGTH), np.linspace(10, 80, len(sig)), sig)
                    if np.max(sig) > 0: sig = 100.0 * (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
                    if is_x: arr[i, :, 0] = sig
                    else: arr[i, :] = sig
                except:
                    if is_x: arr[i, :, 0] = np.zeros(TARGET_LENGTH)
                    else: arr[i, :] = np.zeros(TARGET_LENGTH)
        return X, y

def main():
    with open(MAPPING_PATH, 'rb') as f: id_to_ref = pickle.load(f)
    ref_list = sorted([f for f in os.listdir('Novel-Space/References') if f.endswith('.cif')])
    all_cids = sorted(list(id_to_ref.keys()))
    random.seed(SEED); random.shuffle(all_cids)
    split_idx = int(0.99 * len(all_cids))
    train_cids_dict = {cid: id_to_ref[cid] for cid in all_cids[:split_idx]}
    val_cids_dict = {cid: id_to_ref[cid] for cid in all_cids[split_idx:]}
    train_gen = XRDReconstructionGenerator(train_cids_dict, ref_list, batch_size=BATCH_SIZE)
    val_gen = XRDReconstructionGenerator(val_cids_dict, ref_list, batch_size=BATCH_SIZE)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape, UpSampling1D, Input
    model = Sequential([
        Input(shape=(TARGET_LENGTH, 1)),
        Conv1D(32, 100, strides=5, padding='same', activation='relu'),
        MaxPooling1D(2, padding='same'),
        Conv1D(16, 50, strides=5, padding='same', activation='relu'),
        MaxPooling1D(2, padding='same'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(35 * 16, activation='relu'), 
        Reshape((35, 16)),
        UpSampling1D(2),
        Conv1D(16, 50, padding='same', activation='relu'),
        UpSampling1D(5),
        Conv1D(32, 100, padding='same', activation='relu'),
        UpSampling1D(2),
        UpSampling1D(5),
        Conv1D(1, 3, padding='same', activation='linear'),
        Reshape((TARGET_LENGTH,))
    ])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print(f"Starting training. Best model will be saved to {MODEL_OUT_PATH}")
    model.fit(train_gen, validation_data=val_gen, epochs=NUM_EPOCHS, callbacks=callbacks)

if __name__ == "__main__": main()
