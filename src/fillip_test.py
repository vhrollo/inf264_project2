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

seed = 42
INIT_LR = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_SPLIT = 0.70
TEST_SPLIT = 1 - TRAIN_SPLIT



def data(device):
    dataset = np.load('./data/dataset.npz',)
    X, y = dataset['X'], dataset['y']
    X = scale_data10(X)

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, 
        y, 
        test_size=TEST_SPLIT, 
        random_state=seed
        )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, 
        y_val_test, 
        test_size=0.5, 
        random_state=seed
        )

    X_train, y_train = generate_balanced_data(X_train, y_train, 42)

    X_train = X_train.reshape(-1, 1, 20, 20)
    X_val = X_val.reshape(-1, 1, 20, 20)
    X_test = X_test.reshape(-1, 1, 20, 20)

    # this is doen so that the values are on the cpu already
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        worker_init_fn = lambda _: np.random.seed(seed)
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        worker_init_fn = lambda _: np.random.seed(seed)
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        worker_init_fn = lambda _: np.random.seed(seed)
        )

    return train_loader, val_loader, test_loader, y_test


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1) 
    
    train_loader, val_loader, test_loader, y_test = data(device)
    model = LeNet(numChannels=1, classes=17).to(device)
    model = model.float()
    optimizer = torch.optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()

    H = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print("training the network now")

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()

        total_train_loss = 0
        total_val_loss = 0

        correct_train = 0
        correct_val = 0

        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train += (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()
            # print(y_pred)
            # print("//")

        with torch.no_grad():
            model.eval()

            for X_val_batch, y_val_batch in val_loader:
                y_val_pred = model(X_val_batch)
                loss = criterion(y_val_pred, y_val_batch)
                total_val_loss += loss.item()
                correct_val += (y_val_pred.argmax(1) == y_val_batch).type(torch.float).sum().item()
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        train_acc = correct_train / len(train_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)

        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)
        H['train_acc'].append(train_acc)
        H['val_acc'].append(val_acc)

        end = time.time()

        print(f"--- Epoch {epoch+1}/{EPOCHS} - time: {end-start:.2f}s ---")
        print(f"Train loss: {avg_train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        print(f"Val loss: {avg_val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        print()

    print("training complete") 

    with torch.no_grad():
        model.eval()
        preds = []
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            preds.append(y_pred.argmax(1))

        preds = torch.cat(preds).cpu().numpy()

    print(classification_report(y_test.cpu().numpy(), preds))






if __name__ == "__main__":
    main()
    