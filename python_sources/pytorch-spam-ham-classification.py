#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import torch.nn.functional as F


# In[ ]:


data = pd.read_csv("../input/SMSSpamCollection.tsv", delimiter='\t', header=None, names=["outcome", 'message'])


# In[ ]:


data.head()


# In[ ]:


data.outcome = data.outcome.map({'ham':0, 'spam':1})


# In[ ]:


data.head()


# In[ ]:


features = data.message.values
labels = data.outcome.values
num_words = 1000


# In[ ]:


features.shape


# In[ ]:


labels.shape


# In[ ]:


t = Tokenizer(num_words=1000)
t.fit_on_texts(features)


# In[ ]:


features = t.texts_to_matrix(features, mode='tfidf')


# In[ ]:


features.shape


# In[ ]:


# Building model
class Model(nn.Module):
    def __init__(self, input, hidden, output):
        super(Model, self).__init__()
        self.l1 = nn.Linear(input, hidden)
        self.l2 = nn.Linear(hidden , hidden)
        self.l3 = nn.Linear(hidden, 2)
    
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out        


# In[ ]:


input = 1000
hidden=100
output = 2


# In[ ]:


model = Model(input, hidden, output)


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, shuffle=True, random_state=34)


# In[ ]:


# params
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# In[ ]:


def train(epochs):
    x_train = Variable(torch.from_numpy(features_train)).float()
    y_train = Variable(torch.from_numpy(labels_train)).long()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        print ("epoch #",epoch)
        print ("loss: ", loss.item())
        pred = torch.max(y_pred, 1)[1].eq(y_train).sum()
        print ("acc:(%) ", 100*pred/len(x_train))
        loss.backward()
        optimizer.step()


# In[ ]:


def test(epochs):
    model.eval()
    x_test = Variable(torch.from_numpy(features_test)).float()
    y_test = Variable(torch.from_numpy(labels_test)).long()
    for epoch in range(epochs):
        with torch.no_grad():
            y_pred = model(x_test)
            loss = criterion(y_pred, y_test)
            print ("epoch #",epoch)
            print ("loss: ", loss.item())
            pred = torch.max(y_pred, 1)[1].eq(y_test).sum()
            print ("acc (%): ", 100*pred/len(x_test))


# In[ ]:


train(100)


# In[ ]:


test(100)


# In[ ]:


pred = model(torch.from_numpy(features_test).float())


# In[ ]:


pred


# In[ ]:


pred = torch.max(pred,1)[1]


# In[ ]:


len(pred)


# In[ ]:


len(features_test)


# In[ ]:


pred = pred.data.numpy()


# In[ ]:


pred


# In[ ]:


labels_test


# In[ ]:


accuracy_score(labels_test, pred)


# In[ ]:


p_train = model(torch.from_numpy(features_train).float())


# In[ ]:


p_train = torch.max(p_train,1)[1]


# In[ ]:


len(p_train)


# In[ ]:


p_train = p_train.data.numpy()


# In[ ]:


p_train


# In[ ]:


accuracy_score(labels_train, p_train)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm = confusion_matrix(labels_test, pred)


# In[ ]:


print (cm)


# In[ ]:




