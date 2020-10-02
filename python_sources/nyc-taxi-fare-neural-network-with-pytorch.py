#!/usr/bin/env python
# coding: utf-8

# ## This is a simple neural network to solve NYC Taxi Fare Challenge using Pytorch
# 
# Most of the data preprocessing code is taken from this kernel [Simple Linear Model
# ](https://www.kaggle.com/dster/nyc-taxi-fare-starter-kernel-simple-linear-model)
# 
# Input of the network, similar to the Simple Linear Model kernel, is just the traveling distance of the taxi
# 
# The network used:
# model = nn.Sequential(nn.Linear(2, 10),
#                      nn.Linear(10, 5),
#                       nn.Linear(5, 1))

# In[ ]:


import numpy as np
import pandas as pd
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = '../input'


# In[ ]:


os.listdir(PATH)


# In[ ]:


train_df = pd.read_csv(f'{PATH}/train.csv', nrows=10000000)


# **Remove null data from the dataframe**

# In[ ]:


print(train_df.isnull().sum())
print('Old size %d'% len(train_df))
train_df = train_df.dropna(how='any',axis='rows')
print('New size %d' % len(train_df))


# In[ ]:


train_df[:5]


# **Add new features: travelling distance**

# In[ ]:


def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)


# In[ ]:


train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# **Remove the bizzare travelling distance **

# In[ ]:


train_df = train_df[(train_df.abs_diff_longitude<5) & (train_df.abs_diff_latitude<5)]
print(len(train_df))


# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable


# In[ ]:


model = nn.Sequential(nn.Linear(2, 10),
                     nn.Linear(10, 5),
                      nn.Linear(5, 1))
                    


# In[ ]:


criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# In[ ]:


X = np.stack((train_df.abs_diff_latitude.values,train_df.abs_diff_longitude.values)).T
X = torch.from_numpy(X)
X = X.type(torch.FloatTensor)


# In[ ]:


y = torch.from_numpy(train_df.fare_amount.values.T)
y = y.type(torch.FloatTensor)
y.unsqueeze_(-1)


# **Initial training with high learning rate 0.01**

# In[ ]:


for epoch in range(90):
    # Forward Propagation
    y_pred = model(X)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()


# **Further training with lower learning rate lr=0.001**

# In[ ]:


optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# In[ ]:


for epoch in range(700):
    # Forward Propagation
    y_pred = model(X)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()


# In[ ]:


y[:10]


# In[ ]:


y_pred[:10]


# In[ ]:


test_df = pd.read_csv(f'{PATH}/test.csv')


# In[ ]:


test_df


# In[ ]:


add_travel_vector_features(test_df)


# In[ ]:


X_test = np.stack((test_df.abs_diff_latitude.values,test_df.abs_diff_longitude.values)).T
X_test = torch.from_numpy(X_test)
X_test = X_test.type(torch.FloatTensor)


# In[ ]:


y_test = model(X_test)


# In[ ]:


y_test[:20]


# In[ ]:


test_df.key


# In[ ]:


y_test = y_test.detach().numpy()


# In[ ]:


y_test = y_test.reshape(-1)


# In[ ]:


y_test


# In[ ]:


submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


# In[ ]:


print(os.listdir('.'))


# ## Conclusion
# 
# This approach give me the score about 5.76538 and top 80%. 
# 
# **Improvement**
# First improvement I think is adding the nonlinear layer in the network. I tried to use RELU after each linear layer except the last layer but the result is worst. I'm very appreciate if someone can point out how to add some nonlinear activate function to improve the score.
# 
