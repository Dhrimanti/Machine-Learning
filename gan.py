import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm as tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data  

(trainX, trainy), (testX, testy) = load_data()
trainX=(np.float32(trainX)-127.5)/127.5