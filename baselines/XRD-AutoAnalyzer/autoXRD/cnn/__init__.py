import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.keras import regularizers
import sys

# Used to apply dropout during training *and* inference
class CustomDropout(tf.keras.layers.Layer):

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": self.rate
        })
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class DataSetUp(object):
    """
    Class used to train a convolutional neural network on a given
    set of X-ray diffraction spectra to perform phase identification.
    """

    def __init__(self, xrd, testing_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by
                their associated reference phase.
                The shape of the array should be NxMx3500x1 where:
                N = the number of reference phases,
                M = the number of augmented spectra per reference phase,
                3500 = intensities as a function of 2-theta
                (spanning from 10 to 80 degrees by default)
            testing_fraction: fraction of data (xrd patterns) to reserve for testing.
                By default, all spectra will be used for training.
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)

    @property
    def phase_indices(self):
        """
        List of indices to keep track of xrd spectra such that
            each index is associated with a reference phase.
        """
        xrd = self.xrd
        num_phases = self.num_phases
        return [v for v in range(num_phases)]

    @property
    def x(self):
        """
        Feature matrix (array of intensities used for training)
        """
        xrd = self.xrd
        # Determine total samples and pattern length
        total_samples = sum(len(aug) for aug in xrd)
        
        # Find the first non-empty augmented spectra to get shape
        pattern_shape = None
        for aug in xrd:
            if len(aug) > 0:
                pattern_shape = np.array(aug[0]).shape
                break
        
        if pattern_shape is None:
            return np.array([])

        # Pre-allocate numpy array to save memory
        intensities = np.zeros((total_samples, *pattern_shape), dtype=np.float32)
        
        curr_idx = 0
        for augmented_spectra in xrd:
            num_aug = len(augmented_spectra)
            if num_aug > 0:
                intensities[curr_idx:curr_idx+num_aug] = np.array(augmented_spectra, dtype=np.float32)
                curr_idx += num_aug
        
        # Ensure channel dimension exists for Conv1D
        if len(intensities.shape) == 2:
            intensities = np.expand_dims(intensities, axis=-1)
            
        return intensities

    @property
    def y(self):
        """
        Target property to predict (indices associated
        with the reference phases) for sparse categorical crossentropy
        """
        xrd = self.xrd
        total_samples = sum(len(aug) for aug in xrd)
        
        # Pre-allocate numpy array
        indices = np.zeros(total_samples, dtype=np.int32)
        
        curr_idx = 0
        for index, augmented_spectra in enumerate(xrd):
            num_aug = len(augmented_spectra)
            for _ in range(num_aug):
                indices[curr_idx] = index
                curr_idx += 1
        return indices

    def split_training_testing(self):
        """
        Training and testing data will be split according
        to self.testing_fraction

        Returns:
            train_x, train_y, test_x, test_y: training/testing datasets
        """
        # x/y properties already pre-allocate and use float32/int32
        x = self.x
        y = self.y
        testing_fraction = self.testing_fraction
        
        total_samples = len(x)
        # Use a random permutation of indices
        indices = np.random.permutation(total_samples)
        
        # Shuffle in-place if possible or use fancy indexing
        x = x[indices]
        y = y[indices]

        # Explicitly delete indices after use
        del indices

        if testing_fraction == 0:
            return x, y, None, None

        else:
            n_testing = int(testing_fraction*total_samples)
            
            # Use slicing to create views/copies
            test_x = x[:n_testing].copy()
            test_y = y[:n_testing].copy()
            
            train_x = x[n_testing:].copy()
            train_y = y[n_testing:].copy()

            # Clear large arrays to free memory before returning
            del x
            del y

            return train_x, train_y, test_x, test_y

def train_model(x_train, y_train, n_phases, num_epochs, is_pdf, n_dense=[3100, 1200], dropout_rate=0.7):
    """
    Args:
        x_train: numpy array of simulated xrd spectra
        y_train: one-hot encoded vectors associated with reference phase indices
        n_phases: number of reference phases considered
        fmodel: filename to save trained model to
        n_dense: number of nodes comprising the two hidden layers in the neural network
        dropout_rate: fraction of connections excluded between the hidden layers during training
    Returns:
        model: trained and compiled tensorflow.keras.Model object
    """

    # Optimized architecture for PDF analysis
    if is_pdf:
        model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=60, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_phases, activation='softmax')])

    # Optimized architecture for XRD analysis
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=35, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=30, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=25, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=20, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=15, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_phases, activation='softmax')])

    # Compile model
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Fit model to training data
    # Use smaller batch_size to save memory
    model.fit(x_train, y_train, batch_size=128, epochs=num_epochs,
    validation_split=0.2, shuffle=True)

    return model

def test_model(model, test_x, test_y):
    """
    Args:
        model: trained tensorflow.keras.Model object
        x_test: feature matrix containing xrd spectra
        y_test: one-hot encoded vectors associated with
            the reference phases
    """
    _, acc = model.evaluate(test_x, test_y)
    print('Test Accuracy: ' + str(acc*100) + '%')

def main(xrd, num_epochs, testing_fraction, is_pdf, fmodel='Model.h5'):

    # Organize data
    obj = DataSetUp(xrd, testing_fraction)
    num_phases = obj.num_phases
    train_x, train_y, test_x, test_y = obj.split_training_testing()

    # Train model
    model = train_model(train_x, train_y, num_phases, num_epochs, is_pdf)

    # Save model
    model.save(fmodel, include_optimizer=False)

    # Test model is any data is reserved for testing
    if testing_fraction != 0:
        test_model(model, test_x, test_y)
