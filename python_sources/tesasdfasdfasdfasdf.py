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


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F         

torch.manual_seed(1)


# In[ ]:


train_data=pd.read_csv('../input/regression-cabbage-price/train_cabbage_price.csv',header=None,skiprows=1, usecols=range(1,6))
train_data


# In[ ]:





# In[ ]:





# In[ ]:


device = torch.device("cuda") 

GPU_x_data=train_data.loc[:,1:4]
GPU_y_data=train_data.loc[:,5]

GPU_x_data=np.array(GPU_x_data)
GPU_y_data=np.array(GPU_y_data)

GPU_x_data=torch.FloatTensor(GPU_x_data).to(device)
GPU_y_data=torch.FloatTensor(GPU_y_data).to(device)

print(GPU_x_data.shape)
print(GPU_y_data.shape)


# In[ ]:


GPU_W=torch.zeros((4,1)).to(device).detach().requires_grad_(True)
GPU_b=torch.zeros(1).to(device).detach().requires_grad_(True)


GPU_optimizer=optim.SGD([GPU_W,GPU_b],lr=0.00001)

nb_epochs=1000

for epoch in range(nb_epochs+1):
  GPU_hypothesis=GPU_x_data.matmul(GPU_W)+GPU_b
  GPU_cost=torch.mean((GPU_hypothesis-GPU_y_data)**2)
  
  GPU_optimizer.zero_grad()
  GPU_cost.backward()
  GPU_optimizer.step()

  if epoch %100==0:
    print('Epoch {:4d}/{}  Cost {:.6f}'.format(epoch,nb_epochs, GPU_cost.item()))


# In[ ]:


test_data=pd.read_csv('../input/regression-cabbage-price/test_cabbage_price.csv',header=None,skiprows=1,usecols=range(1,5))
test_data


# In[ ]:


test=np.array(test_data)
test=torch.FloatTensor(test).to(device)

print(test)


# In[ ]:


predict=test.matmul(GPU_W)+GPU_b
print(predict)


# In[ ]:


submit=pd.read_csv('../input/regression-cabbage-price/submit_sample.csv')
submit


# In[ ]:


for i in range(len(predict)):
  submit['Expected'][i]=predict[i].item()
submit


# In[ ]:


submit.to_csv('results-jwkim.csv',index=False,header=True)


# In[ ]:




