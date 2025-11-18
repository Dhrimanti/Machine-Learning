import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

sns.set_theme()

(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255
testX = np.float32(testX) / 255

class Adam:
    def __init__(self,model,alpha=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.model=model
        self.mt=[torch.zeros_like(p) for p in model.parameters()]
        self.vt=[torch.zeros_like(p) for p in model.parameters()]
        self.t=0
        self.beta1=beta1
        self.beta2=beta2
        self.alpha=alpha
        self.epsilon=epsilon
    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad=torch.zeros_like(p.grad)
    def step(self):
        self.t += 1
        b1 = self.beta1
        b2 = self.beta2
        lr = self.alpha
        eps = self.epsilon
        t = self.t

        for i, p in enumerate(self.model.parameters()):
            if p.grad is None:
                continue
            g = p.grad.data
            self.mt[i] = b1 * self.mt[i] + (1 - b1) * g
            self.vt[i] = b2 * self.vt[i] + (1 - b2) * (g * g)

            m_hat = self.mt[i] / (1 - b1 ** t)
            v_hat = self.vt[i] / (1 - b2 ** t)

            with torch.no_grad():
                p.data = p.data - lr * m_hat / (v_hat.sqrt() + eps)

def train(model,optimizer,loss_fct=torch.nn.NLLLoss(),nb_epochs=5,batch_size=128):
    testing_accuracy=[]
    for epoch in range(nb_epochs):
        indices=torch