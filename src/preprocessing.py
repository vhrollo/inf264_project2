import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import random

# preprocessing functions

def generate_label(X: np.ndarray, y: np.ndarray, label: int, n: int, seed: Optional[int] = None):
    """
    Generates n augmented samples for the given label\\
    using the ImageDataGenerator from Keras.
    """

    # get the data for the given label
    X = X[y == label]
    y = y[y == label]

    # set seed for reproducibility
    if seed is not None: 
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    # augment the data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
    )
    X_reshaped = X.reshape(X.shape[0], 20, 20, 1)

    # generate n augmented samples
    augmented_data = datagen.flow(X_reshaped, y, batch_size=1, seed=seed)

    # store the augmented data
    X_augs, y_augs = [], []
    for i in range(n):
        X_aug, y_aug = augmented_data.__next__()
        X_aug = X_aug.flatten()
        X_augs.append(X_aug)
        y_augs.append(y_aug)
    
    # convert to numpy arrays
    X_augs = np.array(X_augs)
    y_augs = np.array(y_augs).reshape(-1)

    return np.array(X_augs), np.array(y_augs)


def generate_balanced_data(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    generates balanced data by augmenting the minority classes\\
    to match the majority class. The minority classes are\\
    augmented by rotating, shifting, shearing, and zooming the images.\\
    The augmentation is done using the ImageDataGenerator from Keras.

    Parameters:
        X (np.ndarray):
            The input data
        y (np.ndarray):
            The labels
        seed (int):
            The seed for reproducibility

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            The balanced data
    """

    # set seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    labels, counts = np.unique(y, return_counts=True)
    n_samples = np.max(counts)
    X_balanced, y_balanced = X.copy(), y.copy()

    # augment the minority classes to match the majority class
    for label in labels:
        if counts[label] < n_samples:
            X_aug, y_aug = generate_label(X, y, label, n_samples - counts[label], seed)
            X_balanced = np.vstack([X_balanced, X_aug])
            y_balanced = np.hstack([y_balanced, y_aug])

    return X_balanced, y_balanced


def scale_data(X: np.ndarray) -> np.ndarray:
    """
    scales the data using Sklearn StandardScaler
    """
    ss = StandardScaler()
    return ss.fit_transform(X)


def scale_data10(X: np.ndarray) -> np.ndarray:
    """
    scales the data to be between 0 and 1
    """
    return X / 255


def hog_features(X: np.ndarray) -> np.ndarray:
    """
    Extracts HOG features from the images.
    """
    hog_features = []
    for img in X:
        hog_feat = hog(img.reshape(20, 20), 
                       orientations=9, 
                       pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), 
                       block_norm='L2-Hys', 
                       visualize=False)
        hog_features.append(hog_feat)
    return np.array(hog_features)


def color_cutoff(data: np.ndarray, black_cutoff, white_cutoff) -> np.ndarray:
        """
        Thresholds the data to black and white\\
        if the pixel value is below black_cutoff\\ 
        or above white_cutoff
        """
        b_filter = np.where(data < black_cutoff, 0, data)
        w_filter = np.where(b_filter > white_cutoff, 255, b_filter)
        return w_filter