import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm



class MLP(nn.module):
    def __init__(self,in_dim,context_dim,h,out_dim):
        super(MLP,self).__init__()
        self.network=nn.Sequential(nn.Linear(in_dim+context_dim,h),nn.Tanh(),
                                   nn.Linear(h,h),
                                   nn.Tanh(),
                                   nn.Linear(h,out_dim))
        def forward(self,x,context):
            return self.network(torch.cat((x,context),dim=1))
class Dataset(torch.utils.data.Dataset):
    def __init__(self,dist1,dist2):
        self.dist1=dist1
        self.dist2=dist2
        assert self.dist1.shape==self.dist2.shape

    def __len__(self):