import matplotlib.pyplot as plt

def visualize_samples(X, y, num_samples=9, image_size=(20, 20)):
    """visualizes the first num_samples samples from the dataset"""
    plt.figure(figsize=(6, 6))
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i+1)
        plt.imshow(X[idx].reshape(image_size), cmap='gray')
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()