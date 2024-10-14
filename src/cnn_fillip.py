import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import time
import random

from preprocessing import generate_balanced_data, scale_data10
from nets import LeNet, AlexNetEsque

# Constants and settings
seed = 42
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_SPLIT = 0.7
VAL_TEST_SPLIT = 0.5
best_model_tuned_path = './other/alexnet_best_model_tuned.pth'
MODEL = AlexNetEsque  # Change to AlexNetEsque if needed
FOLDS = 3

# Data preparation function
def data(device):
    """Prepare data for training and evaluation"""
    dataset = np.load('./data/dataset.npz')
    X, y = dataset['X'], dataset['y']
    X = scale_data10(X)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, train_size=TRAIN_SPLIT, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=VAL_TEST_SPLIT, random_state=seed)


    X_train, y_train = generate_balanced_data(X_train, y_train, seed)

    X_train, X_val, X_test = [torch.tensor(data.reshape(-1, 1, 20, 20), dtype=torch.float32).to(device)
                              for data in [X_train, X_val, X_test]]
    y_train, y_val, y_test = [torch.tensor(labels, dtype=torch.long).to(device)
                              for labels in [y_train, y_val, y_test]]

    train_loader = create_data_loader(X_train, y_train, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, shuffle=False)
    test_loader = create_data_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader


def create_data_loader(X, y, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, worker_init_fn=lambda _: np.random.seed(seed))

# Training function without early stopping
def train_and_evaluate(model, train_loader, device, optimizer, criterion, scheduler, epochs=EPOCHS):
    """Train and evaluate the model"""
    model.train()
    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0
        correct_train = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch.to(device))
            loss = criterion(y_pred, y_batch.to(device))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train += (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_acc = correct_train / len(train_loader.dataset)

        scheduler.step(avg_train_loss)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        end = time.time()
        #print(f"--- Epoch {epoch+1}/{epochs} - time: {end-start:.2f}s ---")
        #print(f"Train loss: {avg_train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        #print()

    return history

# Grid search function for learning rate
def grid_search_learning_rate(X_train, y_train, device):
    """Grid search for the best learning rate and momentum"""
    learning_rates = [0.1, 0.01]
    momentum_rates = [0.9, 0.95]
    best_lr = None
    best_momentum = None
    best_val_acc = 0

    combinations = [(lr, momentum) for lr in learning_rates for momentum in momentum_rates]

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)

    for lr, moementum in combinations:
        print(f"\nEvaluating learning rate: {lr} and momentum: {moementum}")
        val_accs = []

        #changed so that we dont use val data, to avoid data leakage
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"Fold {fold+1}")
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
            y_train_fold = torch.tensor(y_train_fold, dtype=torch.long).to(device)
            X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
            y_val_fold = torch.tensor(y_val_fold, dtype=torch.long).to(device)

            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_fold, y_train_fold), batch_size=BATCH_SIZE, shuffle=True)

            model = MODEL(numChannels=1, classes=17).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moementum)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

            train_and_evaluate(model, train_loader, device, optimizer, criterion, scheduler)

            val_accs.append((model(X_val_fold).argmax(1) == y_val_fold).float().mean().item())

        avg_val_acc = np.mean(val_accs)
        print(f"Learning rate: {lr}, Momentum rate: {moementum}, Average Val accuracy: {avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_lr = lr
            best_momentum = moementum

    print(f"Best learning rate: {best_lr}, Best momentum Rate: {best_momentum} with Val accuracy: {best_val_acc:.4f}")
    return best_lr, best_momentum

# Final evaluation function
def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the validation or test set"""
    model.eval()
    total_loss = 0
    correct = 0
    preds = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch.to(device))
            loss = criterion(y_pred, y_batch.to(device))
            total_loss += loss.item()
            correct += (y_pred.argmax(1) == y_batch).sum().item()
            preds.append(y_pred.argmax(1).cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    return avg_loss, accuracy, preds, labels

# Main function for training and evaluation
def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    train_loader, val_loader, test_loader = data(device)

    # Grid search for the best learning rate
    X_train, y_train = train_loader.dataset.tensors
    print("Starting grid search for the best learning rate and momentum...")
    start = time.time()
    best_lr, best_momentum = grid_search_learning_rate(X_train.cpu().numpy(), y_train.cpu().numpy(), device)
    end = time.time()
    print(f"Grid search took {end-start:.2f}s")
    print(f"Best learning rate: {best_lr}")
    print(f"Best momentum: {best_momentum}")

    # Train final model with the best learning rate
    model = MODEL(numChannels=1, classes=17).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=best_lr, momentum=best_momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    train_and_evaluate(model, train_loader, device, optimizer, criterion, scheduler)

    # Final evaluation on validation and test sets
    print("Evaluating model on validation data...")
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
    print(classification_report(val_labels, val_preds))

    print("Evaluating model on test data...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds))

    torch.save(model.state_dict(), best_model_tuned_path)
    print(f"Final model saved as {best_model_tuned_path}")

if __name__ == "__main__":
    main()

def test_pytorch_cnn(net = 'lenet'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    if net == 'lenet':
        best_model_tuned_path = './other/LeNet_model_tuned.pth'
        MODEL = LeNet
    else:
        best_model_tuned_path = './other/alexnet_best_model_tuned.pth'
        MODEL = AlexNetEsque

    _, val_loader, test_loader = data(device)

    model = MODEL(numChannels=1, classes=17).to(device)
    model.load_state_dict(torch.load(best_model_tuned_path))
    criterion = nn.CrossEntropyLoss()

    print("Best model loaded...")
    print("Evaluating model on validation data...")

    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
    print(classification_report(val_labels, val_preds))

    print("Evaluating model on test data...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(classification_report(test_labels, test_preds))
