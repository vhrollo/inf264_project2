import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

def visualize_confusion_matrix(y_val, y_val_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def compare_model_accuracy(*model_final_acc):
    # Generate model names based on the number of accuracies provided
    models = [f'Model {i+1}' for i in range(len(model_final_acc))]
    accuracies = list(model_final_acc)
    
    plt.figure(figsize=(8, 6))
    
    # Create a bar chart with dynamic colors
    colors = plt.cm.tab10.colors  # Use a colormap with enough distinct colors
    bars = plt.bar(models, accuracies, color=colors[:len(models)])
    
    # Add value labels on top of each bar
    for bar, accuracy in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + 0.005, 
            f'{yval:.2%}', 
            ha='center', 
            va='bottom'
        )
    
    plt.title('Final Model Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1) 
    plt.show()

