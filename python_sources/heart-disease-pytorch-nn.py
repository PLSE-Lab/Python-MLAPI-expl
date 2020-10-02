#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use("ggplot")


# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


df.head()


# In[ ]:


X = df.drop("target", axis=1)
y = df['target']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


sc = MinMaxScaler((-1, 1))


# In[ ]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


import torch
from torch import nn, optim
from torch.nn import functional as F


# In[ ]:


X_train.shape


# In[ ]:


X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train.values).long()
y_test = torch.tensor(y_test.values).long()


# In[ ]:


class HeartDiseaseNN(nn.Module):
    def __init__(self):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(13, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# In[ ]:


net = HeartDiseaseNN()


# In[ ]:


optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()


# In[ ]:


losses = []


# In[ ]:


for epoch in range(1, 201):
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print("Epoch {}, Loss: {}".format(epoch, loss.item()))


# In[ ]:


plt.plot(losses)


# In[ ]:


pred_test = net(X_test)
_, preds_y = torch.max(pred_test, 1)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


accuracy_score(y_test, preds_y)


# In[ ]:




