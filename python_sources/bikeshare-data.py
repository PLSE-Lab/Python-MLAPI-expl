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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


bike_data = pd.read_csv("../input/bike_share.csv")


# In[ ]:


bike_data.info()


# In[ ]:


bike_data.head()


# In[ ]:


bike_data.corr()


# In[ ]:


sns.pairplot(data=bike_data[["season","workingday","temp","atemp","windspeed","count"]])


# In[ ]:


bike_data.season.value_counts().plot(kind="pie")


# From the above observation,the variables holiday,weather and humidity are negatively correlated with count.
# From plot,we could see that, all 4 season values are mostly equal against count.
# Casual and registered variables sum is the value of count variable.
# So bulding the model with variables,season,workingday,temp,atemp and windspeed.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


y = bike_data["count"]
x = bike_data.drop(columns=["season","workingday","holiday","weather","humidity","casual","registered","count"])


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.3,random_state = 42)


# In[ ]:


model=LinearRegression()
model.fit(train_x,train_y)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


train_predict = model.predict(train_x)


# In[ ]:


test_predict = model.predict(test_x)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE - Train :" ,mean_squared_error(train_y,train_predict))
print("MSE - Test :" ,mean_squared_error(test_y,test_predict))
print("MAE - Train :" ,mean_absolute_error(train_y,train_predict))
print("MAE - Test :" ,mean_absolute_error(test_y,test_predict))
print("R2 - Train :" ,r2_score(train_y,train_predict))
print("R2 - Test :" ,r2_score(test_y,test_predict))
print("Mape - Train:" , np.mean(np.abs((train_y,train_predict))))
print("Mape - Test:" ,np.mean(np.abs((test_y,test_predict))))


# In[ ]:




