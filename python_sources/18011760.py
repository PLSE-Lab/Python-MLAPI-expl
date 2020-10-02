#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


get_ipython().system('pip uninstall kaggle')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install kaggle==1.5.6')


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle')
get_ipython().system('ls -lha kaggle.json')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle competitions download -c 2020-ai-exam-fashionmnist-1')


# In[ ]:


get_ipython().system('unzip 2020-ai-exam-fashionmnist-1.zip')


# In[ ]:


import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random

from sklearn import preprocessing


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# In[ ]:


learning_rate = 0.001
training_epochs = 15
batch_size = 100
Scaler = preprocessing.StandardScaler()


# In[ ]:


train_data=pd.read_csv('mnist_train_label.csv',header=None, usecols=range(0,785))
test_data=pd.read_csv('mnist_test.csv',header=None, usecols=range(0,784))


# In[ ]:


x_train=train_data.loc[:,1:785]
y_train=train_data.loc[:,0]

x_train=np.array(x_train)
y_train=np.array(y_train)
x_train = Scaler.fit_transform(x_train)
x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)


# In[ ]:


train_dataset = torch.utils.data.TensorDataset(x_train, y_train)


# In[ ]:


data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)


# In[ ]:


linear1 = torch.nn.Linear(784,10,bias=True)


# In[ ]:


torch.nn.init.xavier_uniform_(linear1.weight)


# In[ ]:


model = torch.nn.Sequential(linear1).to(device)


# In[ ]:


loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 


# In[ ]:



total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:

        X = X.view(-1, 28 * 28).to(device)

        Y = Y.to(device)

        optimizer.zero_grad()

        hypothesis = model(X)

        cost = loss(hypothesis, Y)

        cost.backward()

        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')


# In[ ]:



with torch.no_grad():

  x_test_data=test_data.loc[:,:]
  x_test_data=np.array(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)

  prediction = model(x_test_data)
  correct_prediction = torch.argmax(prediction, 1)


# In[ ]:


correct_prediction = correct_prediction.cpu().numpy().reshape(-1,1)


# In[ ]:


submit=pd.read_csv('submission.csv')
submit


# In[ ]:


for i in range(len(correct_prediction)):
  submit['Category'][i]=correct_prediction[i].item()


# In[ ]:


submit.to_csv('submission.csv',index=False,header=True)

get_ipython().system('kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submission.csv -m "Message"')

