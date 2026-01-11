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

class Generator(nn.Module):
    def __init__(self,input_dim=100,hidden_dim=1200,output_dim=28*28):
        super(Generator,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
            nn.Tanh(),

        )
    def forward(self,noise):
        return self.network(noise)

class Discriminator(nn.Module):
    def __init__(self,input_dim=28*28,hidden_dim=1200,output_dim=1):
        super(Discriminator,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Li
        )
    def forward(self,x):
        return self.network(x)
    
def train(generator,discriminator,generator_optimizer,discriminator_optimizer,nb_epoch,k=1,batch_size=100):
    training_loss={'generative':[],"discriminative":[]}
    for i in tqdm(range(nb_epoch)):
        for i in range(k):
            z=sample_noise(batch_size)
            x=get_minibatch(batch_size)
            f_loss=torch.nn.BCELoss()(discriminator(generator(z))).reshape(batch_size),torch.zeros(batch_size)
            r_loss=torch.nn.BCELoss()(discriminator(x).reshape(batch_size),torch.ones(batch_size))
            loss=(r_loss+f_loss)/2
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            training_loss['discriminative'].append(loss.item())
if  __name__=="__main__":
    discriminator=Discriminator()
    generator=Generator()
    optimizer_d=optim.SGD


