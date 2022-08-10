import torch
import torch.nn as nn
from torch.nn import functional as F

class Multi_net(nn.Module):
    def __init__(self,ind_net,pred_net,feature_dim):
        super(Multi_net,self).__init__()
        self.ind_net = ind_net
        self.pred_net = pred_net
        for p in self.parameters():
            p.requires_grad=False
        self.fc = nn.Sequential(
            nn.Linear(feature_dim,512),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256,3),
        )
    
    def forward(self,x):
        self.ind_net.eval()
        self.pred_net.eval()
        out_i = self.ind_net.extract_feature(x[0])
        out_p = self.pred_net.extract_feature(x[1])
        out_i = out_i.flatten(start_dim=1)
        out_p = out_p.flatten(start_dim=1)
        feature = torch.cat((out_i,out_p),1)
        logits = self.fc(feature)
        return logits
