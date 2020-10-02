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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


bike_df = pd.read_csv("../input/bike_share.csv")


# In[ ]:


bike_df.shape


# In[ ]:


bike_df.head()


# In[ ]:


bike_df.isna().sum()


# In[ ]:


bike_df.corr()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


model = LinearRegression()


# In[ ]:


y = bike_df[["count"]]
x = bike_df.drop(columns = ["count","holiday","workingday","windspeed"])


# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(x,y,test_size=0.3)


# In[ ]:


model.fit(train_X,train_Y)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


train_predict = model.predict(train_X)


# In[ ]:


test_predict = model.predict(test_X)


# In[ ]:


# MSE for train data
print("MSE", mean_squared_error(train_Y,train_predict))


# In[ ]:


# MSE for test data
print("MSE for test", mean_squared_error(test_Y, test_predict))


# In[ ]:


# MAE for train data
print("MAE for train", mean_absolute_error(train_Y,train_predict))


# In[ ]:


# MAE for test data
print("MAE for train", mean_absolute_error(test_Y,test_predict))


# 
