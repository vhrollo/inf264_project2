import matplotlib.pyplot as plt
import numpy as np

def visualize_samples(X, y = None, image_size=(20, 20)):
    """visualizes the first num_samples samples from the dataset"""
    plt.figure(figsize=(6, 6))
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        plt.imshow(X[idx].reshape(image_size), cmap='gray')
        if y is not None:
            plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_preds(X, cols=10):
    num_ood_samples = len(X)

    num_cols = cols
    num_rows = int(np.ceil(num_ood_samples / num_cols))

    _, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))

    for i, ax in enumerate(axes.flat):
        if i < num_ood_samples:
            ax.imshow(X[i].reshape(20, 20), cmap='gray')
            ax.set_title(f"OOD {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()