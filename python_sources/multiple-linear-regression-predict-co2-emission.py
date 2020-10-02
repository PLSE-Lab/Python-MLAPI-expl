#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print('Setup Completed')


# In[ ]:


dataset = "../input/FuelConsumptionCo2.csv"
df_data = pd.read_csv(dataset)


# In[ ]:


df_data.head()


# In[ ]:


df_data.shape


# In[ ]:


df_data.describe()


# In[ ]:


df_data.corr()


# In[ ]:


work_df = df_data[["ENGINESIZE","CYLINDERS","CO2EMISSIONS"]]


# In[ ]:


work_df.head()


# In[ ]:


work_df.corr()


# In[ ]:


msk = np.random.rand(len(work_df)) < 0.8
train_set = work_df[msk]
test_set = work_df[~msk]

print('Training Set Shape : ', train_set.shape)
print('Testing Set Shape : ', test_set.shape)

train_x = np.asanyarray(train_set[["ENGINESIZE","CYLINDERS"]])
train_y = np.asanyarray(train_set[["CO2EMISSIONS"]]).flatten()


# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(train_x,train_y)


# In[ ]:


r_sq = model.score(train_x,train_y)
intercept = model.intercept_
slope = model.coef_

print('r square : ', r_sq)
print('Intercept : ', intercept)
print('Slope : ', slope)


# In[ ]:


test_x = np.asanyarray(test_set[["ENGINESIZE","CYLINDERS"]])
test_y = np.asanyarray(test_set[["CO2EMISSIONS"]]).flatten()


# In[ ]:


y_pred = model.predict(test_x)
print(y_pred)


# In[ ]:


print(test_set)


# In[ ]:


r_sq_test = model.score(test_x,test_y)
print(r_sq_test)


# In[ ]:


mse = np.mean((test_y - y_pred)**2)
print('Mean Squeard Error : ', mse)


# In[ ]:




