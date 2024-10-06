import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, numChannels, classes):
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
