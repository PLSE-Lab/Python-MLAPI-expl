#!/usr/bin/env python
# coding: utf-8

# Hey, This Notebook is an attempt to use rnn network to try to predict frequently bought object together. I am not sure rnn are supposed to be used this way or not but it helped me to learn bout rnn module and also nlp without preprocessing the sentences before being fed to network.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# So we whave loaded basics modules pd and np. Lets load our data into a dataframe

# In[ ]:


df=pd.read_csv('/kaggle/input/online-retail-data-set-from-ml-repository/retail_dataset.csv')


# Our Data Looks Like...

# In[ ]:


df.head()


# lets change nan to <<nothing>>.its not a compulsory but when testing model it seems better to seem "nothing"(ie customer will not take anything else) rather than nan 

# In[ ]:


df.fillna('<nothing>',inplace=True)


# so we need to create a one hot code for our input.
# 
# first we should know all labels.
# 
# we use set to remove all repeated elements

# In[ ]:


labels=df.values.reshape(-1)
labels=set(labels)


# So we have following classes or labels

# In[ ]:


labels        


# Lets One Hot Code them, Later we will also normalize the input so we dont get same result continuously due to screwed data

# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


ohe=OneHotEncoder()


# In[ ]:


ohe.fit(np.asarray(list(labels)).reshape(-1,1))


# so our one hot encoder is ready

# In[ ]:


ohe.categories_


# lets try it out

# In[ ]:


ohe.transform([[df['0'][4]]]).toarray()


# seems fine,we will implement a function later to convert it back to class.

# In[ ]:


import torch as T
import torch.nn as nn


# Lets Create Our Model. It is a simple model with a Gated RNN and then 1 connected layers and 1 output layer

# In[ ]:


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Network,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True) 
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU())
        self.out=nn.Sequential(
                    nn.Linear(hidden_dim,output_size),
                    nn.LogSoftmax()
        
        )
    def forward(self, x):
        
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out=self.out(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        hidden = T.zeros(self.n_layers, batch_size, self.hidden_dim)
    
        return hidden 


# In[ ]:


model=Network(input_size=len(labels),output_size=len(labels),hidden_dim=8,n_layers=1)


# In[ ]:


model


# Lets try our model

# In[ ]:


model.forward(T.from_numpy(ohe.transform([[df['0'][0]]]).toarray()).unsqueeze(0).float())


# here first tensor is our output or we can use these to extract prob for each classes later in our function predict()

# Lets us create X(input) for our model,first we take one hot encodings of columns of df and then concatenate it to previously calculated ohe of previous columns 

# In[ ]:


X=np.zeros((1,7,10))

for x in range(len(df)):
    X=np.concatenate((X,ohe.transform(df.values[x].reshape(-1, 1)).toarray().reshape(1,7,10)),axis=0)


# In[ ]:


X.shape


# lets try X[0] our model

# In[ ]:


model.forward(T.from_numpy(X[1]).float().unsqueeze(0))


# okay so first input is actually all zeros we will not convert them to tensors

# In[ ]:


X=T.from_numpy(X[1:]).float()


# In[ ]:


X


# We are using Adams optimizer and cross entropy as our loss function

# In[ ]:


optimizer=T.optim.Adam(model.parameters())


# In[ ]:


creterion = nn.CrossEntropyLoss()


# Lets check our loss and optimizer for one step

# In[ ]:


out=model.forward(X[0].unsqueeze(0))


# In[ ]:


out


# Choosing Y. we our using crossentropy which requires [0 to num_classes] as target and probs as y_hat

# SO Y is actually next input or X, so we shift X one unit and get max to get idx of class to which it belong

# In[ ]:


Y=X[:,1:,:].max(axis=2)[1]


# similarly there is no use of last value in x as a input

# In[ ]:


X=X[:,0:-1,:]


# Lets check our model and solve errors if any

# In[ ]:


out=model.forward(X[0].unsqueeze(0))


# In[ ]:


X[:,0:,:].shape


# In[ ]:


creterion(out[0],Y[0])


# so result is fine for one output, lets check for whole dataset

# In[ ]:


X.shape


# In[ ]:


out=model.forward(X)


# In[ ]:


creterion(out[0],Y.view(-1,1).squeeze(1))


# so model  is ready,
# 
# A final step:
# see we dont wnat our model predicting too much <nothing> class. We we count all occurences of a word and then assign a low value to <nothing> class.

# In[ ]:


#normalizing
s=df['0'].value_counts()+df['1'].value_counts()+df['3'].value_counts()+df['2'].value_counts()+df['4'].value_counts()+df['5'].value_counts()+df['6'].value_counts()


# In[ ]:


s=2*s


# In[ ]:


s[0]=50


# In[ ]:


s


# In[ ]:


s=6*s/sum(s)


# In[ ]:


s


# In[ ]:


mask=T.from_numpy(s.to_numpy())


# In[ ]:


mask


# In[ ]:


X=X*mask


# In[ ]:


losses=[]
for i in range(6500):
        
   optimizer.zero_grad()
   out=model.forward(X.float())
   loss=creterion(out[0],Y.view(-1,1).squeeze(1))
        
   losses.append(loss.item())
   loss.backward()
   if(i%500==0):
        print('loss is {}'.format(loss.item()))
   optimizer.step()


# In[ ]:


from matplotlib import pyplot as plt
plt.plot(losses)


# So we have reduced our loss and curve seems to flat out

# lets do some predictions

# In[ ]:


def predict(model,item):
#     item=ohe.transform(item)
    out, hidden = model(item)
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    idx = T.topk(prob,k=4, dim=0)
    return idx, hidden


# lets check our fucntion

# In[ ]:


p=predict(model,X[2,0].reshape([1,1,-1]).float())


# So we are actually getting top 4 classes with highest probilities of comming next.

# In[ ]:


p[0][1]


# This is our function to convert back ohe to class ame

# In[ ]:


def conv_from_idx(idx):
    return ohe.categories_[0][idx]


# In[ ]:


conv_from_idx(p[0][1][0])


# Meat is having highest probability to be a good suggestion depepnding upon first thing consumer buy

# In[ ]:


from random import randint


# In[ ]:


def sample(model,inital_data):
    l=[]
    for i in inital_data:

        p=predict(model,i.float())
        print(conv_from_idx(p[0][1][0]))
        l.append(p[0])
    return l


# In[ ]:


X.shape


# So lets predict the muliple results

# In[ ]:


for i in range(100):
    l = []
    for j in range(3):
        l.append(conv_from_idx(X[i][j].max(axis=0)[1]))
        print('for {} consumer can buy '.format(l),end=' ')
        sample(model,[X[i][0:j+1].unsqueeze(0).float()])


# So we need a diaper after buying bread and wine :D

# In[ ]:




