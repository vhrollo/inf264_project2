import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocessing import generate_balanced_data, hog_features
from visiualising import visualize_confusion_matrix

from time import time
import random

def data(X, y, seed=42):
    """Prepare data for training and evaluation"""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=0.7, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed
    )

    # balancing the data with augmentation pieces
    X_train, y_train = generate_balanced_data(X_train, y_train, seed)

    # extracting hog features
    X_train = hog_features(X_train)
    X_val = hog_features(X_val)
    X_test = hog_features(X_test)

    # scaling the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_rf(X_train, y_train, n, folds, seed=42):
    """Train the RandomForestClassifier model with Bagging"""

    # optimized for time and performance
    param_dist = {
        'n_estimators': np.random.randint(100, 300, size=50),
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None] + list(np.arange(10, 30, 10)),
        'min_samples_split': np.random.randint(2, 6, size=50),
        'min_samples_leaf': np.random.randint(1, 6, size=50),
        'bootstrap': [True, False],
    }

    # training the model with random search
    rf = RandomForestClassifier(random_state=seed)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n,
        cv=folds, n_jobs=1, verbose=2, random_state=seed, scoring='accuracy'
    )

    # training the model
    print("---------- training RandomForest model ----------")
    start = time()
    random_search.fit(X_train, y_train)
    end = time()

    print("---------- training is finished. ----------\n")
    print(f"Random Search took {end - start:.2f} seconds")
    print(f"Best Parameters: {random_search.best_params_}")

    best_rf = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_rf, best_params


def evaluate(model, X, y):
    """Evaluate the model on the validation and test set"""
    y_pred = model.predict(X)

    c_r = classification_report(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    f1 = f1_score(y, y_pred, average='weighted')

    return y_pred, accuracy, c_r, f1


def main():
    # mainly used for testing
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    dataset = np.load('./data/dataset.npz')
    X, y = dataset['X'], dataset['y']

    X_train, X_val, X_test, y_train, y_val, y_test = data(X, y, seed=seed)

    best_rf, best_params = train_rf(X_train, y_train, n=5, folds=3, seed=seed)

    print("Evaluating...")
    print(f"Best Parameters: {best_params}\n")
    print("\n--Evaluating on Validation Set--")

    y_val_pred, val_acc, val_c_r, val_f1= evaluate(best_rf, X_val, y_val)
    print(val_c_r)
    print(f"Validation Accuracy: {val_acc:.4f}")
    visualize_confusion_matrix(y_val, y_val_pred, title='Confusion Matrix on Validation Set')

    print("\n--Evaluating on Test Set--")
    y_test_pred, test_acc, test_c_r, test_f1 = evaluate(best_rf, X_test, y_test)
    print(test_c_r)
    print(f"Test Accuracy: {test_acc:.4f}")
    visualize_confusion_matrix(y_test, y_test_pred, title='Confusion Matrix on Test Set')


if __name__ == "__main__":
    main()

# making this into a class for convenience
class RandomForest:
    def __init__(self, X, y, n, folds, seed=42):
        """
        Initialize the RandomForest model with the data and parameters.
        Just added to make our svm isnt lonely surrounded by only
        state of the art models CNN's.
        
        Parameters:
            X (np.ndarray):
                The input data
            y (np.ndarray):
                The labels
            n (int):
                The number of iterations for the RandomizedSearchCV
            folds (int):
                The number of folds for the cross-validation
            seed (int):
                The seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        X_train, X_val, X_test, y_train, y_val, y_test = data(X, y, seed=seed)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.n = n
        self.folds = folds

    def fit(self):
        """Fit the RandomForest model on the training data"""
        self.best_model, self.best_params = train_rf(
            self.X_train, 
            self.y_train, 
            self.n, 
            self.folds,
            self.seed)

    def evaluate(self):
        """Evaluate the model on the validation and test set"""
        y_val_pred, val_acc, val_c_r, val_f1 = evaluate(self.best_model, self.X_val, self.y_val)
        return (y_val_pred, val_acc, val_c_r, val_f1,self.y_val)

    def test(self):
        """Evaluate the model on the test set"""
        y_test_pred, test_acc, test_c_r, test_f1 = evaluate(self.best_model, self.X_test, self.y_test)
        return (y_test_pred, test_acc, test_c_r, test_f1, self.y_test)
