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
        return self.dist1.shape[0]
    def __getitem__(self,idx):
        return self.dist1[idx],self.dist2[idx]
    
def sample_multimodal_distribution(modes, std, batch_size=1000):
    dataset = []
    for i in range(batch_size):
        sample = np.random.randn(modes.shape[1]) * std
        mode_idx = np.random.randint(modes.shape[0])
        sample[0] += modes[mode_idx, 0]
        sample[1] += modes[mode_idx, 1]
        dataset.append(sample)
    return np.array(dataset, dtype="float32")


def train_rectified(rectified_flow,optimizer,train_loader,NB_EPC,eps=1e-15):
    for i in tqdm(range(NB_EPC)):
        for z0,z1 in (train_loader):
            t=torch.rand((z1.shape[0],1))
            z_t=t*z1+(1.-t)*z0
            target=z1-z0
            pred=rectified_flow(z_t,t)
            loss=(target-pred).view(pred.shape[0],-1).abs().pow(2).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()