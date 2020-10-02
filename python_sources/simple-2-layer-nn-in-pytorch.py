#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
print(os.listdir("../input"))


# Read the training and test[](http://) datasets

# In[11]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')\n\nprint(train.shape, test.shape)")


# Split the training dataset for training and validation

# In[12]:


y = train['target'].values
X = train.drop(['ID_code', 'target'], axis=1).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


print(len(X_train), len(X_val))
print(len(y_train), len(y_val))


# Construct a 2-Layer NN

# In[14]:


#Seed
torch.manual_seed(1234)

#hyperparameters
hl = 10
lr = 0.01
num_epoch = 100

#Model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(200, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


# Train the NN

# In[15]:


get_ipython().run_cell_magic('time', '', "#train\nfor epoch in range(num_epoch):\n    X = Variable(torch.Tensor(X_train).float())\n    Y = Variable(torch.Tensor(y_train).long())\n\n    #feedforward - backprop\n    optimizer.zero_grad()\n    out = net(X)\n    loss = criterion(out, Y)\n    loss.backward()\n    optimizer.step()\n\n    if (epoch) % 10 == 0:\n        print ('Epoch [%d/%d] Loss: %.4f' \n                   %(epoch+1, num_epoch, loss.item()))")


# Validate the NN

# In[16]:


get_ipython().run_cell_magic('time', '', "\n#Validation\nX = Variable(torch.Tensor(X_val).float())\nY = torch.Tensor(y_val).long()\nout = net(X)\n\n_, predicted = torch.max(out.data, 1)\n\n#get accuration\nprint('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / len(y_val)))")


# Perform prediction on test dataset

# In[17]:


get_ipython().run_cell_magic('time', '', "\n#Test\nX_test = test.drop(['ID_code'], axis=1).values\n\nX = Variable(torch.Tensor(X_test).float())\nout = net(X)\n\n_, predicted = torch.max(out.data, 1)")


# Output prediction to CSV

# In[18]:


ID_code = test['ID_code']
target = predicted.data.numpy()

my_submission = pd.DataFrame({'ID_code': ID_code, 'target': target})
my_submission.to_csv('submission.csv', index=False)

my_submission.head()


# In[ ]:




