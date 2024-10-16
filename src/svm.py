import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from preprocessing import generate_balanced_data, hog_features, color_cutoff
from visiualising import visualize_preds, visualize_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

    X_train = hog_features(X_train)
    X_val = hog_features(X_val)
    X_test = hog_features(X_test)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train(X_train, y_train, n, folds, seed=42):
    """train the model without pca"""
    uniform_samples = 100
    param_dist = {
        'C': np.random.uniform(0.1, 100, size=uniform_samples),
        'gamma': np.random.uniform(1e-4, 1e-1, size=uniform_samples),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'coef0': np.random.uniform(0, 1, size=uniform_samples),
        'shrinking': [True, False],
    }

    svc = SVC(random_state=seed, probability=True)
    random_search = RandomizedSearchCV(
        svc, param_distributions=param_dist, n_iter=n,
        cv=folds, n_jobs=1, verbose=2, random_state=seed, scoring='accuracy'
    )
    start = time()
    random_search.fit(X_train, y_train)
    end = time() 

    print("---------- training is finished. ----------\n")
    print(f"Random Search took {end - start:.2f} seconds")
    print(f"Best Parameters: {random_search.best_params_}")

    best_svc = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_svc, best_params

def train_pca(X_train, y_train, X_val, X_test, n, folds, pca_n=5, seed=42):
    """train the model with pca where pca is a hyperparameter as well"""

    n_components_list = np.linspace(0.9, 0.99, pca_n)

    best_accuracy = 0
    best_n_components = None
    best_pca = None
    best_model = None

    uniform_samples = 100
    param_dist = {
        'C': np.random.uniform(0.1, 100, size=uniform_samples),
        'gamma': np.random.uniform(1e-4, 1e-1, size=uniform_samples),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4],
        'coef0': np.random.uniform(0, 1, size=uniform_samples),
        'shrinking': [True, False],
    }

    from itertools import product
    from sklearn.model_selection import GridSearchCV
    params_list = list(product(param_dist['C'], param_dist['gamma'], param_dist['kernel'], param_dist['degree'], param_dist['coef0'], param_dist['shrinking']))
    param_list_shorted = random.sample(params_list, n) # sample n random parameters

    # convert to list of dictionaries
    param_list_shorted = [{
        'C': [p[0]], 
        'gamma': [p[1]], 
        'kernel': [p[2]], 
        'degree': [p[3]], 
        'coef0': [p[4]], 
        'shrinking': [p[5]]
        }
        for p in param_list_shorted
    ]

    print("Starting PCA training...")
    pca_start = time()
    for i, n_components in enumerate(n_components_list):
        print(f"\n--Training with {n_components:.2f} components {i+1}/{pca_n}--")

        pca = PCA(n_components=n_components, random_state=seed)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        
        print(f"Data went from {X_train.shape} to {X_train_pca.shape}")

        svc = SVC(random_state=seed, probability=True)

        
        grid_search = GridSearchCV(
            svc, param_grid=param_list_shorted,
            cv=folds, n_jobs=1, verbose=2, scoring='accuracy'
        )

        start = time()
        grid_search.fit(X_train_pca, y_train)
        end = time()
        print(f"Random Search for {n_components:.2f} took {end - start:.2f} s")

        best_svc = grid_search.best_estimator_
        best_params = grid_search.best_params_
        accuracy = grid_search.best_score_

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_components = n_components
            best_pca = pca
            best_model = best_svc

    pca_end = time()
    print("---------- training is finished. ----------\n")
    print(f"PCA training took {pca_end - pca_start:.2f} seconds")
    print(f"Best Parameters: {best_params} and best n_components: {best_n_components}")

    return best_model, best_params, best_pca, best_n_components, X_val_pca, X_test_pca


def evaluate(model,X, y):
    """Evaluate the model on the validation and test set"""
    y_pred = model.predict(X)

    c_r = classification_report(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return y_pred, accuracy, c_r


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    dataset = np.load('./data/dataset.npz')
    X, y = dataset['X'], dataset['y']

    X_train, X_val, X_test, y_train, y_val, y_test = data(X, y, seed=seed)

    best_svc, best_params = train(X_train, y_train, seed=seed)
    
    print("Evaluating...")
    print(f"Best Parameters: {best_params}\n")
    print("\n--Evaluating on Validation Set--")

    y_val_pred, val_acc, val_c_r = evaluate(best_svc, X_val, y_val)
    print(val_c_r)
    print(f"Validation Accuracy: {val_acc:.4f}")
    visualize_confusion_matrix(y_val, y_val_pred, title='Confusion Matrix on Validation Set')

    print("\n--Evaluating on Test Set--")
    y_test_pred, test_acc, test_c_r = evaluate(best_svc, X_test, y_test)
    print(test_c_r)
    print(f"Test Accuracy: {test_acc:.4f}")
    visualize_confusion_matrix(y_test, y_test_pred, title='Confusion Matrix on Test Set')


if __name__ == "__main__":
    main()

# making this into a class for convenience
class SVM:
    # this abstraction of a class is horrible i really am sorry lmao
    def __init__(self, X, y, n, folds, pca_n=5, pca=False, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.uses_pca = pca
        X_train, X_val, X_test, y_train, y_val, y_test = data(X, y, seed=seed)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.n = n
        self.pca_n = pca_n
        self.folds = folds

    def fit(self):
        if self.uses_pca:
            #felt very cursed writing this mb
            self.best_model, self.best_params, self.best_pca, self.best_n_components, self.X_val_pca, self.X_test_pca = train_pca(
                self.X_train, 
                self.y_train, 
                self.X_val, 
                self.X_test,
                self.n,
                self.folds,
                self.pca_n,
                self.seed)
        else:
            self.best_model, self.best_params = train(
                self.X_train, 
                self.y_train,
                self.n,
                self.folds, 
                self.seed)

    def evaluate(self):
        if self.uses_pca:
            y_val_pred, val_acc, val_c_r = evaluate(self.best_model, self.X_val_pca, self.y_val)
            return (y_val_pred, val_acc, val_c_r)
        else:
            y_val_pred, val_acc, val_c_r = evaluate(self.best_model, self.X_val, self.y_val)
            return (y_val_pred, val_acc, val_c_r)
        
    def test(self):
        if self.uses_pca:
            y_test_pred, test_acc, test_c_r = evaluate(self.best_model, self.X_test_pca, self.y_test)
            return (y_test_pred, test_acc, test_c_r)
        else:
            y_test_pred, test_acc, test_c_r = evaluate(self.best_model, self.X_test, self.y_test)
            return (y_test_pred, test_acc, test_c_r)