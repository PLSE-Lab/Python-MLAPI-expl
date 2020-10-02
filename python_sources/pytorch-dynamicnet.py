#!/usr/bin/env python
# coding: utf-8

# ## Example Illustrating Dynamic Computational Graphs

# Load the torch packages

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


# ### NN model
# * 2 Layer network
# * One hidden and one output layer
# * Computation Graphs are constructed on each forward pass
# * Each forward pass chooses a random number between 1 and 4(excluding) and uses that many hidden layers.

# In[3]:


class DynamicNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_layer = nn.Linear(D_in, H)  #input layer
        self.middle_layer = nn.Linear(H, H)  #hidden layer
        self.output_layer = nn.Linear(H, D_out)  #output layer
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        #choose random number between 1 and 4
        for _ in range(random.randint(0,4)):
            x = F.relu(self.middle_layer(x))
        y = self.output_layer(x)
        return y


# ### Train
# * Traing the model for 500 iterations on randomly chosen input and outputs.
# * Plot the losses.

# In[4]:


import matplotlib.pyplot as plt

N, D_in, D_out, H = 64, 1000, 10, 100

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9)

fig = plt.figure()
losses = list()
for t in range(500):
    y_pred = model(x)
    
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t = list(t for t in range(500))
plt.plot(t, losses, 'r-')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

