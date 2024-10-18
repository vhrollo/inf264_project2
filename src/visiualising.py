import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_samples(X, y = None, image_size=(20, 20)):
    """
    visualizes the first num_samples samples from the dataset.
    If y is provided, it will also display the label for each sample.
    Given that hog will be used, and it will resample to another size,
    we added image size by deafult 20x20.
    """
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


def visualize_preds(X, title,cols=10):
    """will just visualize the all in X, with being 'cols' wide'"""

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

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_confusion_matrix(y_val, y_val_pred, title='Confusion Matrix'):
    """
    Visualize the confusion matrix of the model
    using seaborn's heatmap, bc it looks better B-).
    """
    cm = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def compare_model_accuracy(header, labels, *model_final_acc):
    """
    Compare the accuracy of different models,
    which works with any amount of models and 
    their accuracies.
    """
    # Generate model names based on the number of accuracies provided
    length = max(len(model_final_acc) - len(labels), 0)
    models = labels[:len(model_final_acc)] + [f'Model {length + i+1}' for i in range(length)]
    
    accuracies = list(model_final_acc)

    max_acc = max(accuracies)
    min_acc = min(accuracies)    


    plt.figure(figsize=(8, 6))
    
    # Create a bar chart with dynamic colors
    colors = plt.cm.Pastel1.colors
    bars = plt.bar(models, accuracies, color=colors[:len(models)])
    
    # Add value labels on top of each bar
    for bar, accuracy in zip(bars, accuracies):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + 0.01, 
            f'{yval:.2%}', 
            ha='center', 
            va='bottom'
        )
    
    x_positions = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
    plt.plot(x_positions, accuracies, color='coral', marker='o', linestyle='-', label='Accuracy Trend')


    plt.title(f"{header}")
    plt.ylabel('Accuracy')
    padding = 0.05
    plt.ylim(min_acc - padding, max_acc + padding)
    plt.show()


def visualize_common_wrong_guesses(header, y_true, X_train, *false_guesses):
    """visualize the common wrong guesses of the model"""
    # indeces where all models got it wrong
    
    wrong_indices_sets = []
    
    for model_preds in false_guesses:
        wrong_indices = set(np.where(model_preds != y_true)[0]) 
        wrong_indices_sets.append(wrong_indices)
    
    common_wrong = set.intersection(*wrong_indices_sets)

    print(f"Common wrong indices: {common_wrong}")

    # visualize the common wrong guesses
    visualize_preds(X_train[list(common_wrong)], title=header)

    return common_wrong