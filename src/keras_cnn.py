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

from nets import build_ANModel, build_LNModel
####
import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set TensorFlow logging to only show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CNN_KERAS:
    def __init__(self, seed, epochs, path, threshold = .85,retrain = False, model = 'lenet'):
        random.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

        self.seed = seed
        self.epochs = epochs
        self.model_path = path
        self.threshold = threshold
        self.retrain = retrain

        if model == 'lenet':
            self.best_model_tuned_path = f"{path}LeNet_model_tuned.keras"
            self.MODEL = build_LNModel
            self.name = "LeNet"
        elif model == 'alexnet':
            self.best_model_tuned_path = f"{path}alexnet_model_tuned.keras"
            self.MODEL = build_ANModel
            self.name = "AlexNet"
        else:
            raise ValueError("Invalid model type")
        
    def fit(self, X, y):
        # use "Leave on Out" validation. Keeps original validation set unseen
        X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=.7, random_state = self.seed)

        #Use keras-tuner for hyperparameter search. Save run to folder "HyperTuningRS".
        # If retrain = True, overwrite earlier run
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
        tunerRandomSearch.search(X_train,y_train, validation_data = (X_val,y_val),epochs=self.epochs,callbacks=[skip_low])
        # Save best hyperparameters
        self.best_hps = tunerRandomSearch.get_best_hyperparameters()[0]

        # Get the best model
        self.best_trained_model = tunerRandomSearch.get_best_models()[0]
        model = self.best_trained_model
        # Fit the best params to the whole training-data
        model.fit(X,y,epochs=8)
        # Save / overwrite the best model as a file in /other
        model.save(self.best_model_tuned_path)
    
    def loadModel(path = "../other/best_model.keras"):
        model = keras.models.load_model(path)
        return model




# Define a callback class to work with keras tuner
class SkipLowAcc (keras.callbacks.Callback):
    #Callback that ends if val_acc is below given threshold
    def __init__ (self, threshold):
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if (val_acc is not None) and (val_acc < self.threshold):
            print(f"ERROR\n current val_acc = {val_acc} < {self.threshold} in epoch {epoch}")
            self.model.stop_training = True
        
