#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import sklearn.datasets
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# **Create a dummy dataset**

# In[ ]:


X, y = sklearn.datasets.make_classification(n_samples=5000, n_features=20, n_informative=20, n_redundant=0, n_repeated=0, n_classes=2)


# ** Standard Train/Test split **

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# ** Convert the samples into Torch Tensors. I used Floats in this instance because of my approach using Binary Cross Entropy loss and a Sigmoid output layer. **

# In[ ]:


X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor)


# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)


# In[ ]:


train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)


# In[ ]:


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


# ** Create the model itself. You need to have function to make all of the different layers (init), a forward function that actually creates the networks architecture (forward), and a predict function to get your answers out (predict). The way I've architected this network I could just call forward instead of predict but I like to keep them seperate. **

# In[ ]:


class NN_Classifier(nn.Module):
    
    def __init__(self):
        super(NN_Classifier, self).__init__()
        self.in_layer = nn.Linear(in_features=20, out_features=40)
        self.hidden_layer = nn.Linear(in_features=40, out_features=1)
        self.out_layer = nn.Sigmoid()
    
    def forward(self, x):
        x = self.in_layer(x)
        x = self.hidden_layer(x)
        x = self.out_layer(x)
        
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        return pred


# ** Instantiate model, determine loss, and optimizer. **

# In[ ]:


model = NN_Classifier()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ** I like to look at the networks architecture, even when it's really simple like this example. **

# In[ ]:


model.eval()


# In[ ]:


epochs = 1000


# In[ ]:


losses = []
for i in range(epochs):
    batch_loss = []
    for x_batch, y_batch in train_loader:
        y_pred = model.forward(x_batch)
        loss = criterion(y_pred, y_batch)
        losses.append(loss.item())
        batch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i%100==0:
        print('Epoch %s: ' % i + str(np.mean(batch_loss)))


# In[ ]:


y_hat = model.predict(X_test)


# In[ ]:


yhat = []
for i in y_hat:
    if i >= .5:
        yhat.append(1)
    else:
        yhat.append(0)


# In[ ]:


print(classification_report(y_test, yhat))


# In[ ]:




