#!/usr/bin/env python
# coding: utf-8

# # Predicting a Pulsar Star using PyTorch
# 
# We will use PyTorch to predict whether the candidate is a pulsar star
# 
# We will import and load the data in the following cells

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt # Importing matplotlib for plotting data
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import seaborn as sns


# In[ ]:


df = pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")


# In[ ]:


df.head()


# ## Exploring the Data.
# We are exploring the data here. We call `df.describe` to see some statistics about the dataset

# In[ ]:


df.describe()


# Splitting data to inputs and labels.

# In[ ]:


inputs = df.drop("target_class", axis=1)
labels = df['target_class']


# In[ ]:


labels.head()


# In[ ]:


labels.hist(bins=2) # We can see that there is not a lot of pulsar stars so the neural network is more likely to predict that something is not a pulsar


# In[ ]:


inputs.head()


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


import torch.optim as optim


# In[ ]:


len(inputs.columns)


# ## Building the Network
# 
# We are initializing the network below. We will use 3 fully connected layers. The first layer has an input shape of 8 because there are 8 fields

# In[ ]:


class PulsarStarNN(nn.Module):
    def __init__(self):
        super(PulsarStarNN, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


# We are splitting the data to training and testing here

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(inputs, labels, test_size=0.25)


# In[ ]:


train_X = torch.tensor(train_X.values).float()


# In[ ]:


train_X


# In[ ]:


test_X = torch.tensor(test_X.values).float()
train_y = torch.tensor(train_y.values).long()
test_y = torch.tensor(test_y.values).long()


# In[ ]:


train_y


# ## Training the Network
# 
# We will train the neural network for 200 epochs. Watch the loss as it decreases

# In[ ]:


net = PulsarStarNN()


# In[ ]:


net


# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# In[ ]:


losses = []


# In[ ]:


for epoch in range(1, 201):
    optimizer.zero_grad()
    outputs = net(train_X)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print("Epoch {}, Loss: {}".format(epoch, loss.item()))


# ## Testing accuracy
# 
# We plot the losses and test the accuracy. It should get over 90%. You can also see that the loss stays the same after 100 epochs

# In[ ]:


plt.plot(losses)


# In[ ]:


pred_test = net(test_X)
_, preds_y = torch.max(pred_test, 1)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[ ]:


accuracy_score(test_y, preds_y)


# In[ ]:


print(classification_report(test_y, preds_y))


# In[ ]:


confusion_matrix(test_y, preds_y)


# In[ ]:




