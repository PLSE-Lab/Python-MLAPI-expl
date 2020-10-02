#!/usr/bin/env python
# coding: utf-8

# ## Houses to rent data
# 
# <hr> 
# 
# <br>**Content:**
# 1. [Load Libraries and Dataset](#1)
# 1. [Feature Engineering](#2)
# 1. [Linear Regression](#3)

# <a id="1"></a> <br>
# ## Load Libraries and Dataset

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew
#
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
import torch.nn as nn 
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
data.head()


# <a id="2"></a> <br>
# ## Feature Engineering

# In[ ]:


data.info()


# In[ ]:


data = data[data.floor != "-"]
data.floor = data.floor.astype(np.int64)


# In[ ]:


data.info()


# ## Dataset Shuffle

# In[ ]:


data = data.sample(frac=1)


# ## Observe Normalization

# In[ ]:


features = ["floor","bathroom","rooms","area","parking spaces",'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)','total (R$)']
skew_list = []
for i in features:
    skew_list.append(skew(data[i]))
# So, features are good at skewness 
skew_list


# ## Log Transformations

# In[ ]:


features = ["floor","bathroom","rooms","area","parking spaces",'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)','total (R$)']
for item in features:
    data[item] = np.log1p(data[item])


# In[ ]:


features = ["floor","bathroom","rooms","area","parking spaces",'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)','total (R$)']
skew_list = []
for i in features:
    skew_list.append(skew(data[i]))
# So, features are good at skewness 
skew_list


# In[ ]:


f,ax = plt.subplots(figsize = (20,7))
sns.distplot(data["total (R$)"], fit=norm);


# ## One-Hot Encoder

# In[ ]:


data = pd.get_dummies(data,drop_first=True)
data.head()


# ## Split Dataset

# In[ ]:


y = data[["total (R$)"]]
x = data.drop(["total (R$)"],axis=1)


# In[ ]:


# Creating Train and Test Datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


sc_x = StandardScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)


# In[ ]:


sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train)
y_test_scaled = sc_y.transform(y_test)


# In[ ]:


x_train_scaled = np.array(x_train_scaled,dtype=np.float32)
y_train_scaled = np.array(y_train_scaled,dtype=np.float32)


# In[ ]:


# Convert inputs and targets to tensors
inputs = torch.from_numpy(x_train_scaled)
targets = torch.from_numpy(y_train_scaled)


# <a id="3"></a> <br>
# ## Linear Regression

# In[ ]:


# create class
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression,self).__init__()
        # Linear function.
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x): # x:inputs
        return self.linear(x)
    
# define model
input_dim = 15
output_dim = 1
model = LinearRegression(input_dim,output_dim) # 

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

# train model
loss_list = []
iteration_number = 500
for iteration in range(iteration_number):
        
    # optimization
    optimizer.zero_grad() 
    
    # Forward to get output
    results = model(inputs)
    
    # Calculate Loss
    loss = mse(results, targets)
    
    # backward propagation
    loss.backward()
    
    # Updating parameters
    optimizer.step()
    
    # store loss
    loss_list.append(loss.data)
    
    # print loss
    if(iteration % 50 == 0):
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()


# ## Prediction

# In[ ]:


input_x_test = torch.from_numpy(x_test_scaled)
predicted = model(input_x_test.float()).data.numpy()

