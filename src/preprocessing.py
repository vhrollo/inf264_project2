import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Optional

def generate_label(X: np.ndarray, y: np.ndarray, label: int, n: int, seed: Optional[int] = None):
    """generates n augmented images for a given label"""
    X = X[y == label]
    y = y[y == label]
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    X_reshaped = X.reshape(X.shape[0], 20, 20, 1)

    augmented_data = datagen.flow(X_reshaped, y, batch_size=1, seed=seed)
    X_augs, y_augs = [], []
    for i in range(n):
        X_aug, y_aug = augmented_data.__next__()
        X_aug = X_aug.flatten()
        X_augs.append(X_aug)
        y_augs.append(y_aug)
    
    X_augs = np.array(X_augs)
    y_augs = np.array(y_augs).reshape(-1)

    return np.array(X_augs), np.array(y_augs)


def generate_balanced_data(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """generates balanced data by augmenting the minority classes"""

    if seed is not None: np.random.seed(seed)

    labels, counts = np.unique(y, return_counts=True)
    n_samples = np.max(counts)
    X_balanced, y_balanced = X.copy(), y.copy()
    for label in labels:
        if counts[label] < n_samples:
            X_aug, y_aug = generate_label(X, y, label, n_samples - counts[label], seed)
            X_balanced = np.vstack([X_balanced, X_aug])
            y_balanced = np.hstack([y_balanced, y_aug])

    return X_balanced, y_balanced