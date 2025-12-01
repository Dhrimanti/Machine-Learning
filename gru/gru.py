import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import tqdm as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

sns.set_theme()

(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255
testX = np.float32(testX) / 255

def preprocess_imdb(batch_size=32, max_length=500):
    tokenizer = get_tokenizer('basic_english')
    
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])
    
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: 1 if x == 'pos' else 0
    
    def collate_batch(batch):
        text_list, label_list = [], []
        for (_label, _text) in batch:
            processed_text = text_pipeline(_text)
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length]
            text_list.append(torch.tensor(processed_text, dtype=torch.long))
            label_list.append(label_pipeline(_label))
        
        text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
        label_list = torch.tensor(label_list, dtype=torch.long)
        
        return text_list, label_list
    
    train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    return train_loader, test_loader, vocab, len(vocab)

train_loader, test_loader, vocab, vocab_size = preprocess_imdb(batch_size=32)

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

    def reset(self):
        stdv=1/np.sqrt(self.hidden_size)
        for i in self.parameters():
            i.data.uniform_(-stdv,stdv)

    def forward(self,x,prev=None):
        if prev is None:
            prev=torch.zeros(x.size(0),self.hidden_size,dtype=x.dtype,device=x.device)
        z = torch.sigmoid(torch.matmul(x, self.W_z.t()) + torch.matmul(prev, self.U_z.t()) + self.b_z)
        r = torch.sigmoid(torch.matmul(x, self.W_r.t()) + torch.matmul(prev, self.U_r.t()) + self.b_r)
        hidden = torch.tanh(torch.matmul(x, self.W_h.t()) + torch.matmul(r * prev, self.U_h.t()) + self.b_h)

        h_last=(1-z)*hidden+z*prev
        return h_last


def train(model,optimizer,loss_fct=torch.nn.NLLLoss(),nb_epochs=500,batch_size=128):
    model.train()
    for i in range(nb_epochs):
        total_loss=0
        for batch_idx,(data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            output=model(data)
            loss=loss_fct(output,target)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
    return total_loss

class Model(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,use_custom=False):
        super(Model,self).__init__()
        self.use_custom=use_custom
        self.hidden_dim=hidden_dim
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        if self.use_custom:
            self.gru=GRUCell(embedding_dim,hidden_dim)
        else:
            self.gru=nn.GRU(embedding_dim,hidden_dim,batch_first=True)
        self.fc=nn.Linear(hidden_dim,2)

    def forward(self,x):
        embedded=self.embedding(x)
        if self.use_custom:
            batch_size=embedded.size()
