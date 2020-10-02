#!/usr/bin/env python
# coding: utf-8

# In[143]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[144]:


import nltk
from tqdm import tqdm_notebook
from collections import Counter
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader, Dataset


# In[145]:


seed = 5143
np.random.seed(seed)
torch.manual_seed(seed)


# In[ ]:





# In[146]:


root_dir = "../input/aclimdb/aclImdb/"
test_dir = root_dir + "test/"
train_dir = root_dir + "train/"
print(os.listdir(test_dir))
print(os.listdir(train_dir))


# In[147]:


vocab = None
vocab_max = 6000
vocab_min = 15
text_len  = 100
with open(root_dir + "imdb.vocab") as f:
    vocab = {w.lower():k for k, w in enumerate(f.read().split()) if k <= vocab_max+vocab_min and k > vocab_min}
print(len(vocab))


# In[148]:


def check_text(path):
    avg_len = 0
    counter = 0
    for p in ("/pos/", "/neg/"):
        files = os.listdir(path + p)
        for c,i in tqdm_notebook(enumerate(files), total=len(files)):
                avg_len += len(open(path + p + i).read().split())
                counter += 1
    print(avg_len/counter)


# In[149]:


#check_text(train_dir)
#check_text(test_dir)


# In[150]:


def getdata(path, vocab, max_len):
    data   = []
    labels = []
    label_val = 1
    for p in ("/pos/", "/neg/"):
        files = os.listdir(path + p)
        for _,i in tqdm_notebook(enumerate(files), total=len(files)):
            labels.append(label_val)
            file = open(path + p + i)
            text = TweetTokenizer().tokenize(file.read().lower())
            tok_text = []
            for word in text:
                tok_word = vocab.get(word)
                if tok_word is not None:
                    tok_text.append(tok_word)
            if len(tok_text) > max_len:
                tok_text = tok_text[:max_len]
            elif len(tok_text) < max_len:
                tok_text += [0 for i in range(max_len-len(tok_text))]
            data.append(tok_text)
            file.close()
        label_val = 0
    return data, labels


# In[151]:


train_x, train_y = getdata(train_dir, vocab, text_len)


# In[152]:


print(train_x[1])


# In[153]:


test_x, test_y = getdata(test_dir, vocab, text_len)


# In[154]:


valid_x, pred_x, valid_y, pred_x = train_test_split(test_x, test_y, test_size=0.7)


# In[155]:


class Textset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
        
    def __getitem__(self, i):
        return torch.tensor(data=self.data[i]), torch.FloatTensor(data=[self.labels[i]])
    
    def __len__(self):
        return len(self.labels)


# In[156]:


trainset = Textset(train_x, train_y)
trainloader = DataLoader(trainset, batch_size=30, shuffle=True)


# In[157]:


validset = Textset(valid_x, valid_y)
validloader = DataLoader(validset, batch_size=30)


# In[158]:


testset = Textset(test_x, test_y)
testloader = DataLoader(testset, batch_size=30)


# In[159]:


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, batch_size, hid_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_size, hid_size, batch_first=True)
        self.lin  = nn.Linear(hid_size, 1)
        
    def forward(self, x):
        x = self.embed(x)
        x, (h_t, h_c) = self.lstm1(x)
        out = self.lin(h_t)
        return torch.sigmoid(out)


# In[160]:


print(max(vocab.values()))


# In[161]:


rnn = RNN(max(vocab.values())+1, 32, 10).cuda()
# +1 - padding(0)


# In[162]:


optim = optimizer.Adam(rnn.parameters(), lr=2e-3)
crit  = nn.BCELoss()


# In[163]:


epoches = 4


# In[164]:


def train(model, trainloader, validloader, epoches, optim,
          crit):
    for epoche in range(epoches):
        rnn.train()
        for c,(xx, yy) in tqdm_notebook(enumerate(trainloader), total=len(trainloader)):
            xx, yy = xx.cuda(), yy.cuda()
            optim.zero_grad()
            out = model(xx)[0]
            loss = crit(out, yy)
            loss.backward()
            optim.step()
            if c % 150 == 0:
                print("Epoche {}    loss= {}".format(epoche, loss.item()))
        rnn.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _,(xx, yy) in tqdm_notebook(enumerate(validloader), total=len(validloader)):
                    y_true.extend(yy.tolist())
                    xx, yy = xx.cuda(), yy.cuda()
                    out = model(xx)[0]
                    y_pred.extend([i[0]>0.5 for i in out.tolist()])
            print(classification_report(y_true, y_pred))


# In[165]:


train(rnn, trainloader, validloader, epoches, optim, crit)


# In[166]:


def predict(model, predloader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):
                y_true.extend(yy.tolist())
                xx, yy = xx.cuda(), yy.cuda()
                out = model(xx)[0]
                y_pred.extend([i[0]>0.5 for i in out.tolist()])
        print(classification_report(y_true, y_pred))
        
    


# In[167]:


predict(rnn, testloader)


# In[305]:


class CNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 50)
        self.c1 = nn.Conv1d(50, 120, 1)#50
        self.p1 = nn.MaxPool1d(5)#10
        self.c2 = nn.Conv1d(120, 300, 1)#10
        self.p2 = nn.MaxPool1d(2)#50
        self.l1 = nn.Linear(3000, 1)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(1,2)
        x = self.c1(x)
        x = torch.relu(self.p1(x))
        x = self.c2(x)
        x = torch.relu(self.p2(x))
        x = x.view(-1,3000)
        x = torch.sigmoid(self.l1(x))
        return x


# In[306]:


cnn = CNN(max(vocab.values())+1).cuda()


# In[307]:


optim_cnn = optimizer.Adam(cnn.parameters(), lr=2e-3)


# In[310]:


def train_cnn(model, trainloader, validloader, epoches, optim,
          crit):
    for epoche in range(epoches):
        rnn.train()
        for c,(xx, yy) in tqdm_notebook(enumerate(trainloader), total=len(trainloader)):
            xx, yy = xx.cuda(), yy.cuda()
            optim.zero_grad()
            out = model(xx)
            loss = crit(out, yy)
            loss.backward()
            optim.step()
            if c % 150 == 0:
                print("Epoche {}    loss= {}".format(epoche, loss.item()))
        rnn.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for _,(xx, yy) in tqdm_notebook(enumerate(validloader), total=len(validloader)):
                    y_true.extend(yy.tolist())
                    xx, yy = xx.cuda(), yy.cuda()
                    out = model(xx)
                    y_pred.extend([i[0]>0.5 for i in out.tolist()])
            print(classification_report(y_true, y_pred))


# In[311]:


train_cnn(cnn, trainloader, validloader, epoches, optim_cnn, crit)


# In[313]:


def predict_cnn(model, predloader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):
                y_true.extend(yy.tolist())
                xx, yy = xx.cuda(), yy.cuda()
                out = model(xx)
                y_pred.extend([i[0]>0.5 for i in out.tolist()])
        print(classification_report(y_true, y_pred))
        
    


# In[315]:


predict_cnn(cnn, testloader)


# In[364]:


def predict_ans(cnn, rnn, predloader):
    cnn.eval()
    rnn.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for _,(xx, yy) in tqdm_notebook(enumerate(predloader), total=len(predloader)):
                y_true.extend(yy.tolist())
                xx, yy = xx.cuda(), yy.cuda()
                out1 = cnn(xx)
                out2 = rnn(xx)[0]
                out = (out1+out2)/2
                y_pred.extend([i[0]>0.5 for i in out.tolist()])
        print(classification_report(y_true, y_pred))
        
    


# In[365]:


predict_ans(cnn, rnn, testloader)

