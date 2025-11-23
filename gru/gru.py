import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
import tqdm as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data


sns.set_theme()

(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255
testX = np.float32(testX) / 255


class GRUCell(nn.Module):
    def __init__(self,input_size,hidden_size,bias=True,device=None):
        super(GRUCell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.bias=bias

        self.W_z=nn.Parameter(torch.Tensor(hidden_size,input_size))
        self.W_r=nn.Parameter(torch.Tensor(hidden_size,input_size))
        self.W_h=nn.Parameter(torch.Tensor(hidden_size,input_size))


        self.U_z=nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.U_r=nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.U_h=nn.Parameter(torch.Tensor(hidden_size,hidden_size))

        if bias:
            self.b_z=nn.Parameter(torch.Tensor(hidden_size))
            self.b_r=nn.Parameter(torch.Tensor(hidden_size))
            self.b_h=nn.Parameter(torch.Tensor(hidden_size))

        else:
            self.register_parameter('b_z',None)
            self.register_parameter('b_r',None)
            self.register_parameter('b_h',None) 

        

