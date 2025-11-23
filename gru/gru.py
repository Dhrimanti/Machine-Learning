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
    def __init__(self,input_size,hidden_size,)