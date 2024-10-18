import keras
import tensorflow as tf
import numpy as np
import random

from sklearn.model_selection import train_test_split, KFold

from keras import backend as K
from keras.optimizers import Adam, SGD
from sklearn.model_selection import GridSearchCV
from keras import layers #Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import keras_tuner as kt

from preprocessing import generate_balanced_data
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from nets import build_ANModel, build_LNModel
####
import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set TensorFlow logging to only show errors (Removed DEBUGGING INFO print)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def data(X, y, seed=42):
    """
    Prepare data for training and evaluation.\\
    
    """
    X_train, X_val_test, y_train, y_val_test = train_test_split(X,y,train_size=0.7,random_state=seed)
    X_val,X_test,y_val,y_test = train_test_split(X_val_test,y_val_test, train_size=0.5, random_state=seed)

    X_train_aug, y_train_aug = generate_balanced_data(X_train, y_train, seed=seed)

    # Reshape. Needed for keras model
    X_train_aug = X_train_aug.reshape(X_train_aug.shape[0], 20, 20, 1)
    X_val = X_val.reshape(X_val.shape[0], 20, 20, 1)
    X_test = X_test.reshape(X_test.shape[0], 20, 20, 1)

    # Scale the data. 
    X_train_aug = X_train_aug / 255
    X_val = X_val / 255
    X_test = X_test / 255

    # Set to categorical. Needed for the keras modle
    y_train_aug = keras.utils.to_categorical(y_train_aug, num_classes=17)
    y_val = keras.utils.to_categorical(y_val, num_classes=17)
    y_test = keras.utils.to_categorical(y_test, num_classes=17)

    return X_train_aug, X_val, X_test, y_train_aug, y_val, y_test



class CNN_KERAS:
    def __init__(self, X, y, seed, epochs, path, threshold = .85,retrain = False, model = 'lenet'):

        random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = data(X, y, seed)
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.seed = seed
        self.epochs = epochs
        self.model_path = path
        self.threshold = threshold
        self.retrain = retrain



        if model == 'lenet':
            self.best_model_tuned_path = f"{path}LeNet_model_tuned.keras"
            self.best_model_tuned_path_new = f"{path}LeNet_model_tuned_new.keras"
            self.MODEL = build_LNModel
            self.name = "LeNet"
        elif model == 'alexnet':
            self.best_model_tuned_path = f"{path}alexnet_model_tuned.keras"
            self.best_model_tuned_path_new = f"{path}alexnet_model_tuned_new.keras"
            self.MODEL = build_ANModel
            self.name = "AlexNet"
        else:
            raise ValueError("Invalid model type")
        

    def fit(self):
        # use "Leave on Out" validation. Keeps original validation set unseen
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(self.X_train, self.y_train, train_size=.7, random_state = self.seed)

        #Use keras-tuner for hyperparameter search. Save run to folder "HyperTuning(NAME)".
        # If retrain = True, overwrite earlier run (if any)

        with tf.device('/CPU:0'):
            tunerRandomSearch = kt.RandomSearch(
                hypermodel = self.MODEL,
                objective = kt.Objective("val_accuracy", "max"),
                seed = self.seed,
                project_name = ("HyperTuning" + self.name),
                overwrite = self.retrain,
                max_trials = 6,
            )

            #Define threshold for early skipping
            skip_low = SkipLowAcc(threshold=self.threshold)
            tunerRandomSearch.search(X_train_split, y_train_split, validation_data = (X_val_split,y_val_split),epochs=self.epochs,callbacks=[skip_low])

            # Save best hyperparameters
            self.best_hps = tunerRandomSearch.get_best_hyperparameters()[0]

            # Get the best model
            self.best_trained_model = tunerRandomSearch.get_best_models()[0]
            model = self.best_trained_model

            # Fit the best params to the whole training data
            model.fit(self.X_train, self.y_train, epochs=8)

            # Save trained model as new file to avoid rewriting original.
            model.save(self.best_model_tuned_path_new)
            self.best_trained_model = model


    def loadModel(self, path = "../other/best_model.keras"):
        model = keras.models.load_model(path)
        return model
    
    
    def evaluate(self, load=False):
        with tf.device('/CPU:0'):
            if load:
                print(f"Loading Pretrained {self.name} Model:\n")
                model = self.loadModel(self.best_model_tuned_path)
            else:
                print(f"Using {self.name} Model Trained Locally:\n")
                model = self.best_trained_model

            # Evaluate on validation data
            loss, acc = model.evaluate(self.X_val, self.y_val)
            print(f"\n {self.name} Model accuracy on validation data : {acc:.4}, with loss:{loss:.4}")

            # Get predictions and classification report
            y_preds = self._predict(model, self.X_val)
            c_r = classification_report(np.argmax(self.y_val, axis=1), y_preds)

            return (y_preds, acc, c_r)
        
    
    def test(self, load=False):
        with tf.device('/CPU:0'):
            if load:
                print(f"Loading Pretrained {self.name} Model:\n")
                model = self.loadModel(self.best_model_tuned_path)
            else:
                print(f"Using {self.name} Model Trained Locally:\n")
                model = self.best_trained_model

            # Evaluate on test set
            loss, acc = model.evaluate(self.X_test, self.y_test)
            print(f"\n {self.name} Model accuracy on test data : {acc:.4}, with loss:{loss:.4}")
        
            # Get predictions and classification report
            y_preds = self._predict(model, self.X_test)
            c_r = classification_report(np.argmax(self.y_test, axis=0), y_preds)

            return (y_preds, acc, c_r)
        
    
    def _predict(self, model, X):
        with tf.device('/CPU:0'):
            y_pred = model.predict(X)
            # need to make a flat list of predicts to make them comparable with the others.
            # Also to plot the results (if needed)

            y_pred = np.argmax(y_pred,axis=1)
            return y_pred
    


# Define a callback class to work with keras tuner
class SkipLowAcc (keras.callbacks.Callback):
    #Callback that ends if val_acc is below given threshold
    def __init__ (self, threshold):
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if (val_acc is not None) and (val_acc < self.threshold):
            print(f"\nERROR\n current val_acc = {val_acc} < {self.threshold} in epoch {epoch}.\n")
            self.model.stop_training = True