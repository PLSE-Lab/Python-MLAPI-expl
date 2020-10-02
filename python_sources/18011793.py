#!/usr/bin/env python
# coding: utf-8

# !pip uninstall kaggle<br>
# !pip install --upgrade pip<br>
# !pip install kaggle==1.5.6<br>
# !mkdir -p ~/.kaggle<br>
# !cp kaggle.json ~/.kaggle<br>
# !ls -lha kaggle.json<br>
# !chmood 600 ~/.kaggle/kaggle.json<br>

# In[ ]:


#!kaggle competitions download -c 2020-ai-exam-fashionmnist-1


# In[ ]:


#!unzip 2020-ai-exam-fashionmnist-1.zip


# In[ ]:


import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

import pandas as pd
import numpy as np


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# In[ ]:


learning_rate = 0.1
training_epochs = 15
batch_size = 100


# In[ ]:


train_data = pd.read_csv('../input/2020-ai-exam-fashionmnist-1/mnist_train_label.csv', header = None)
test_data = pd.read_csv('../input/2020-ai-exam-fashionmnist-1/mnist_test.csv', header = None, usecols=range(1, 785))


# In[ ]:


test_data


# In[ ]:


train_data


# In[ ]:


x_train_data = train_data.loc[:, 1:784]
y_train_data = train_data.loc[:, 0]

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = torch.FloatTensor(x_train_data)
y_train_data = torch.LongTensor(y_train_data)


# In[ ]:


train_dataset =  torch.utils.data.TensorDataset(x_train_data, y_train_data)


# In[ ]:


data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle = True, 
                                          drop_last = True)


# In[ ]:


linear = torch.nn.Linear(784,10, bias = True)


# In[ ]:


torch.nn.init.xavier_normal_(linear.weight)


# In[ ]:


model = torch.nn.Sequential(linear).to(device)


# In[ ]:


loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# In[ ]:


print(len(data_loader))


# In[ ]:


total_batch = len(data_loader)
for epoch in range(training_epochs):
  avg_cost = 0

  for X, Y in data_loader:
    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = loss(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost/total_batch
  print(epoch+1)
  print(avg_cost)
print("f")


# In[ ]:


with torch.no_grad():
  test_data.loc[:, 783] = 0
  x_test = test_data.loc[:,:]
  x_test = np.array(x_test)
  x_test = torch.from_numpy(x_test).float().to(device)
  prediction = model(x_test)
  correct = torch.argmax(prediction, 1)
correct


# In[ ]:


correct = correct.cpu().numpy().reshape(-1, 1)
correct


# In[ ]:


submit = pd.read_csv('../input/2020-ai-exam-fashionmnist-1/submission.csv')
submit


# In[ ]:


for i in range(len(correct)):
  submit['Category'][i] = correct[i].item()
submit


# In[ ]:


submit.to_csv('submit.csv', index = False, header = True)


# In[ ]:


#!kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submit.csv -m "result"

