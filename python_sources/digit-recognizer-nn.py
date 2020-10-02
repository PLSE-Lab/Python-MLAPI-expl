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


import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import torch 
import torchvision
from torchvision import transforms, datasets
from    torch import nn
from    torch.nn import functional as F
from    torch import optim

from    matplotlib import pyplot as plt


# In[ ]:


train = datasets.MNIST("", train=True, download=True,
                      transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                      transform = transforms.Compose([transforms.ToTensor()])) 


# In[ ]:


trainset = [train, batch_size=10, shuffle=True]
testset = [train, batch_size=10, shuffle=True]


# In[ ]:


#batch is useful because patching data through model at once will have a difficult time optimizing. Generalization. Passing through batches, the opitmization with erase overfitting. 
#8 to 64 for batching 
#shuffle to help nn generalize 


# In[ ]:


for data in trainset:
    print(data)
    break


# In[ ]:


x, y = data[0][0], data[1][0]

print(y)


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28,28))
plt.show


# In[ ]:


#balancing (shortening path to decrease loss)optimizer
# 


# In[ ]:


total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,}

for data in trainset: 
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total+=1
        
print(counter_dict)


# In[ ]:


for i in counter_dict:
    print(f"(i): {counter_dict[i]/total*100}")


# In[ ]:


#3.0 Modeling


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F #nn pass parameters (intitalizatiion) 


# In[ ]:


class Net(nn.Module):
    
    def __init__(self):
        super().__init__() #
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #output layer 10 neurons
        
        #feed forward/activation_func --> range between 0,1 contains outputs
    def forward(self, x):  #much variation/logic can be implemented here
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) #contrains single neuron to be fired/ probability distrabution
        return F.log_softmax(x, dim=1) #similar to axis, probability distributuion of batches of tensors.
    

net = Net()
print(net)


# In[ ]:


X = torch.rand((28,28))
X = X.view(-1,28*28) #-1 is tensor of any size 


# In[ ]:


output = net(X)


# In[ ]:


output #actual predictions, now find out loss, forward function  


# In[ ]:


4.0 Loss and optimization


# In[ ]:


import torch.optim as optim  #pass data, adjust loss etc..

optimizer = optim.Adam(net.parameters(), lr=0.001) #used in transfer learning, lr to tell optimizer to step down #decay lr

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featuresets and labels
        X, y = data
        net.zero_grad() #training goes faster, law of diminishing return, zero the gradient, batch training
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y) #2 major ways to calc loss, one hot vector/MSE,
        loss.backward() #backward prop
        optimizer.step() #optimize weights
    print(loss)
        
         


# In[ ]:


correct = 0
total = 0

with torch.no_grad(): #know how good is network, net.train()/net.eval()
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))


# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(X[0].view(28,28))
plt.show


# In[ ]:


print(torch.argmax(net(X[0].view(-1,784))[0])) #prediction/verification


# In[ ]:




