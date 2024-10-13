import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import random

from preprocessing import generate_balanced_data, scale_data10
from nets import LeNet, AlexNetEsque

# if 
seed = 42
BATCH_SIZE = 32
EPOCHS = 10
MOMENTUM = 0.9
EARLY_STOPPING_PATIENCE = 3

TRAIN_SPLIT = 0.70
TEST_SPLIT = 1 - TRAIN_SPLIT

VAL_TEST_SPLIT = 0.5
best_model_path = './other/LeNet_best_model.pth'
best_model_tuned_path = './other/LeNet_best_model_tuned.pth'
MODEL = LeNet # change to AlexNetEsque if needed


def data(device):
    dataset = np.load('./data/dataset.npz')
    X, y = dataset['X'], dataset['y']
    X = scale_data10(X)

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=VAL_TEST_SPLIT, random_state=seed)

    X_train, y_train = generate_balanced_data(X_train, y_train, seed)

    X_train = X_train.reshape(-1, 1, 20, 20)
    X_val = X_val.reshape(-1, 1, 20, 20)
    X_test = X_test.reshape(-1, 1, 20, 20)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda _: np.random.seed(seed))
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=lambda _: np.random.seed(seed))

    return train_loader, val_loader, test_loader, y_test, y_val


def train_and_evaluate(lr, train_loader, val_loader, device):
    print(f"Training with learning rate: {lr}")
    model = MODEL(numChannels=1, classes=17).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    H = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_train_loss = 0
        total_val_loss = 0
        correct_train = 0
        correct_val = 0

        # Training loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train += (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()

        # Validation loop
        with torch.no_grad():
            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                total_val_loss += val_loss.item()
                correct_val += (y_val_pred.argmax(1) == y_val_batch).type(torch.float).sum().item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        train_acc = correct_train / len(train_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)

        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)
        H['train_acc'].append(train_acc)
        H['val_acc'].append(val_acc)

        scheduler.step(avg_val_loss)

        end = time.time()
        print(f"--- Epoch {epoch+1}/{EPOCHS} - time: {end-start:.2f}s ---")
        print(f"Train loss: {avg_train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {avg_val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        print()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

    return best_val_loss, val_acc


def grid_search_learning_rate(train_loader, val_loader, device):
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    best_val_acc = 0
    best_lr = None
    best_model_state = None

    for lr in learning_rates:
        val_loss, val_acc = train_and_evaluate(lr, train_loader, val_loader, device)
        print(f"Learning rate: {lr}, Val accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            best_model_state = torch.load(best_model_path)

    print(f"Best learning rate: {best_lr} with Val accuracy: {best_val_acc:.4f}")
    torch.save(best_model_state, best_model_tuned_path)
    return best_lr


def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1) 

    train_loader, val_loader, test_loader, y_test, y_val = data(device)
    start = time.time()
    best_lr = grid_search_learning_rate(train_loader, val_loader, device)
    end = time.time()
    print(f"Grid search took {end-start:.2f}s")
    model = MODEL(numChannels=1, classes=17).to(device)
    model.load_state_dict(torch.load(best_model_tuned_path))
    criterion = nn.NLLLoss()

    print("Best model loaded...")
    print("best lr: ", best_lr)
    print("Evaluating model...")

    # val
    with torch.no_grad():
        model.eval()

        preds = []
        total_val_loss = 0
        correct_val = 0

        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch)
            total_val_loss += val_loss.item()
            correct_val += (y_val_pred.argmax(1) == y_val_batch).type(torch.float).sum().item()
            preds.append(y_val_pred.argmax(1))

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)

        preds = torch.cat(preds).cpu().numpy()

    print(f"Val loss: {avg_val_loss:.4f}, Val accuracy: {val_acc:.4f}")
    print(classification_report(y_val.cpu().numpy(), preds))
    
    # test
    with torch.no_grad():
        model.eval()

        preds = []
        total_test_loss = 0
        correct_test = 0

        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_test_loss += loss.item()
            correct_test += (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()
            preds.append(y_pred.argmax(1))

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_acc = correct_test / len(test_loader.dataset)

        preds = torch.cat(preds).cpu().numpy()

    print(f"Test loss: {avg_test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(classification_report(y_test.cpu().numpy(), preds))


if __name__ == "__main__":
    main()
