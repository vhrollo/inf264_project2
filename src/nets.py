import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
        """
        LeNet CNN model for the MNIST dataset
        """
        super(LeNet, self).__init__()

        #scales it down to 18x18 x 20
        self.conv1 = nn.Conv2d(
            in_channels=numChannels, 
            out_channels=20,
            kernel_size=(3,3), 
            )
        
        #first relu pass
        self.relu1 = nn.ReLU()
        
        #scales it down to 9x9 x 20
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2)
            )

        #scales it down to 7x7 x 50 
        self.conv2 = nn.Conv2d(
            in_channels=20, 
            out_channels=50,
            kernel_size=(3,3), 
            )

        #second relu pass
        self.relu2 = nn.ReLU()

        #scales it down to 3x3 x 50
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2),
            )

        #takes the 3x3x50 = 450
        self.fc1 = nn.Linear(
            in_features=450,
            out_features=500,
            )
        
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(
            in_features=500,
            out_features=classes,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.logsoftmax(x)
        return output
    


#https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
class AlexNetEsque(nn.Module):
    def __init__(self, numChannels, classes):
        """
        AlexNet CNN model for the MNIST dataset
        """
        super(AlexNetEsque, self).__init__()

        # 20 x 20 x 64
        self.conv1 = nn.Conv2d(
            in_channels=numChannels, 
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()

        # 10 x 10 x 64
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 10 x 10 x 128
        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU()

        # 5 x 5 x 128
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 5 x 5 x 264
        self.conv3 = nn.Conv2d(
            in_channels=128, 
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu3 = nn.ReLU()

        # 5 x 5 x 256
        self.conv4 = nn.Conv2d(
            in_channels=256, 
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu4 = nn.ReLU()

        # 5 x 5 x 128
        self.conv5 = nn.Conv2d(
            in_channels=256, 
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.relu5 = nn.ReLU()

        # 2x2 x 128
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 2x2 x 128 = 512 
        self.fc1 = nn.Linear(
            in_features=512,
            out_features=512
        )
        self.relu6 = nn.ReLU()

        # helps with generalization
        # basically means if you run longer,
        # it gets better
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(
            in_features=512,
            out_features=256
        )
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(
            in_features=256,
            out_features=classes
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _forward_features(self, x):
        """
        helper function to deal for clean implement
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        output = self.logsoftmax(x)
        return output


####################
# Tensorflow Keras #
####################
import keras
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam,SGD

#############
# LeNet CNN #
#############

# Helper function for LeNet style CNN model
def build_LNModel(hparam):
    #optimizer = Adam(learning_rate=learning_rate)
    
    #def hparams
    hparam_n_filter_1 = hparam.Int("Number of Filters (1)", min_value = 128, max_value = 256, step = 2, sampling = "log")
    hparam_n_filter_2 = hparam.Int("Number of Filters (2)", min_value = 256, max_value = 512, step = 2, sampling = "log")
    hparam_learning_rate = hparam.Float("Learning Rate", min_value = 0.001, max_value = .01, step = 10, sampling="log")
    hparam_activation = hparam.Choice("Actrivation function", values=["relu"])
    hparam_units = hparam.Int("Units (dense layer)", min_value = 2048, max_value = 4096, step = 2, sampling = "log")
    hparam_droprate = hparam.Float("Dropout Rate", min_value = .4, max_value = .6, step = .2)    
    hparam_kernel_size = hparam.Choice("Kernel Size", values=[3])
    hparam_momentum = hparam.Choice("Momentum", values=[.89,.99])
    
    
    model = keras.Sequential()
    
    #input layer
    model.add(Convolution2D(filters=hparam_n_filter_1,
                            kernel_size=(3,3),
                            activation=hparam_activation,
                            input_shape = (20,20,1)))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(filters=hparam_n_filter_2,
                            kernel_size=(3,3),
                            activation=hparam_activation))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #flatten output
    model.add(Flatten())
    model.add(Dropout(rate=hparam_droprate))
    model.add(Dense(units=hparam_units, activation=hparam_activation))
    #output layer (softmax for probas)
    model.add(Dense(17,activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=hparam_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

###############
# AlexNet CNN #
###############

def build_ANModel(hparam):

    hparam_n_filter_1 = hparam.Int("Number of Filters (1)", min_value = 128, max_value = 256, step = 2, sampling = "log")
    hparam_n_filter_2 = hparam.Int("Number of Filters (2)", min_value = 256, max_value = 512, step = 2, sampling = "log")
    hparam_learning_rate = hparam.Float("Learning Rate", min_value = 0.001, max_value = .01, step = 10, sampling="log")
    hparam_activation = hparam.Choice("Actrivation function", values=["relu"])
    hparam_units = hparam.Int("Units (dense layer)", min_value = 2048, max_value = 4096, step = 2, sampling = "log")
    hparam_droprate = hparam.Float("Dropout Rate", min_value = .4, max_value = .6, step = .2)
    hparam_pool_size = hparam.Choice("Pool Size", values=[2])
    hparam_kernel_size = hparam.Choice("Kernel Size", values=[3])
    hparam_momentum = hparam.Choice("Momentum", values=[.89,.99])


    model = keras.Sequential()
    #input layer
    model.add(Convolution2D(padding="same", filters=hparam_n_filter_1,kernel_size = (3,3),
                            input_shape = (20,20,1)))

    model.add(MaxPooling2D(pool_size=(hparam_pool_size,hparam_pool_size)))
    model.add(Convolution2D(padding="same", filters=hparam_n_filter_2, kernel_size = (hparam_kernel_size,hparam_kernel_size),
              activation = hparam_activation))
    model.add(MaxPooling2D(pool_size=(hparam_pool_size,hparam_pool_size)))
    
    model.add(Convolution2D(padding="same", filters=hparam_n_filter_1, kernel_size = (hparam_kernel_size,hparam_kernel_size),
              activation = hparam_activation))
    model.add(Convolution2D(padding="same", filters=hparam_n_filter_2, kernel_size = (hparam_kernel_size,hparam_kernel_size),
              activation = hparam_activation))

    model.add(Convolution2D(padding="same", filters=hparam_n_filter_1, kernel_size = (hparam_kernel_size,hparam_kernel_size),
              activation = hparam_activation))
    model.add(MaxPooling2D(pool_size=(hparam_pool_size,hparam_pool_size)))

    model.add(Flatten())
    model.add(Dense(units = hparam_units, activation = hparam_activation))
    model.add(Dropout(rate=hparam_droprate))

    model.add(Dense(units = hparam_units, activation = hparam_activation))
    model.add(Dropout(rate=hparam_droprate))
    
    model.add(Dense(units = hparam_units, activation = hparam_activation))
    model.add(Dropout(rate=hparam_droprate))

    #output layer
    model.add(Dense(17,activation="softmax"))

    model.compile(optimizer=SGD(learning_rate=hparam_learning_rate,momentum=hparam_momentum), loss="categorical_crossentropy", metrics=["accuracy"])
    return model