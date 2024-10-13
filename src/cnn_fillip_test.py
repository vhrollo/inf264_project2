import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
import random
from nets import LeNet, AlexNetEsque
from cnn_fillip import data

# seed B-)
seed = 42

# make sure the paths are correct
best_model_tuned_path_alex = './other/alexnet_best_model_tuned.pth'
best_model_tuned_path_lenet = './other/LeNet_model_tuned.pth'


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
    model = AlexNetEsque(numChannels=1, classes=17).to(device)
    model.load_state_dict(torch.load(best_model_tuned_path_alex))
    criterion = nn.CrossEntropyLoss()

 

    print("Best model loaded...")
    print("Evaluating model...")
    print("\n--------AlexNetEsque-------\n")
    # val for AlexNetEsque
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
    
    # test for AlexNetEsque
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

    #clear
    del model

    model2 = LeNet(numChannels=1, classes=17).to(device)
    model2.load_state_dict(torch.load(best_model_tuned_path_lenet))
    criterion = nn.CrossEntropyLoss()

    print("\n--------LeNet-------\n")
    # val for LeNet
    with torch.no_grad():
        model2.eval()

        preds = []
        total_val_loss = 0
        correct_val = 0

        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = model2(X_val_batch)
            val_loss = criterion(y_val_pred, y_val_batch)
            total_val_loss += val_loss.item()
            correct_val += (y_val_pred.argmax(1) == y_val_batch).type(torch.float).sum().item()
            preds.append(y_val_pred.argmax(1))

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = correct_val / len(val_loader.dataset)

        preds = torch.cat(preds).cpu().numpy()
    
    print(f"Val loss: {avg_val_loss:.4f}, Val accuracy: {val_acc:.4f}")
    print(classification_report(y_val.cpu().numpy(), preds))

    # test for LeNet
    with torch.no_grad():
        model2.eval()

        preds = []
        total_test_loss = 0
        correct_test = 0

        for X_batch, y_batch in test_loader:
            y_pred = model2(X_batch)
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