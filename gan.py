import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm as tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data  

(trainX, trainy), (testX, testy) = load_data()
trainX=(np.float32(trainX)-127.5)/127.5


def get_minibatch(batch_size):
    indices=torch.randperm(trainX.shape[0])[:batch_size]
    return torch.tensor(trainX[indices],dtype=torch.float).reshape(batch_size,-1)

def sample_noise(size,dim=100):
    out=torch.empty(size,dim)
    mean=torch.zeros(size,dim)
    std=torch.ones(dim)
    torch.normal(mean,std,out=out)
    return out
