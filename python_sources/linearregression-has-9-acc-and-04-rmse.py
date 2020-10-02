#!/usr/bin/env python
# coding: utf-8

# # LinearRegression on 80% values has .9 acc and .04 RMSE
# 
# TODO:
#  * add other models

# In[ ]:


get_ipython().system('ls ../input/*csv')


# In[ ]:


files = get_ipython().getoutput('ls ../input/*csv')


# In[ ]:


dataset = files[1]


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


dataset = pd.read_csv(dataset)
dataset[0:5]


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.describe()


# In[ ]:


test = dataset[int(dataset.shape[0]*.8):]
train = dataset[:int(dataset.shape[0]*.8)]


# In[ ]:


train.shape, test.shape


# In[ ]:


train_X = train.iloc[:,1:8]
train_y = train.iloc[:,8:]


# In[ ]:


train_X.shape, train_y.shape


# In[ ]:


test_X = test.iloc[:,1:8]
test_y = test.iloc[:,8:]


# In[ ]:


test_X.shape, test_y.shape


# In[ ]:


from sklearn.linear_model import LinearRegression
import math

model = LinearRegression()
model.fit(train_X, train_y)
pred = model.predict(test_X)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_y, pred))

acc = model.score(test_X, test_y)

print(f'RMSE: {rmse}, ACCURACY: {acc}')


# In[ ]:




