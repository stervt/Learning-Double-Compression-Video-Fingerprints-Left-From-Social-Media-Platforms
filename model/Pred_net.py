import torch.nn.functional as F
import torch.nn as nn
import torch

class Pred_CNN(nn.Module):
    def __init__(self):
        super(Pred_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,5,1,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,5,1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32,32,5,1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,5,1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3,1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Global Average Pooling
            nn.AvgPool2d(16,stride=1,padding=0)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        logits = x
        return logits

class Fc(nn.Module):
    
    def __init__(self):
        super(Fc, self).__init__()
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(256,3)
        )
    def forward(self, x):
        logits = self.fc_net(x)
        return logits
        

class Pred_Net(nn.Module):
    def __init__(self):
        super(Pred_Net, self).__init__()
        self.cnn = Pred_CNN()
        self.fc = Fc()
    def forward(self, x):
        x = self.cnn(x)
        logits = self.fc(x)
        return logits
    def extract_feature(self,x):
        feature = self.cnn(x)
        return feature