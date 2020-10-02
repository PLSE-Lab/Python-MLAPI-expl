#!/usr/bin/env python
# coding: utf-8

# **I Data Loading**

# In[65]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pathlib as Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/train.csv',parse_dates=['pickup_datetime','dropoff_datetime'])
df_test = pd.read_csv('../input/test.csv',parse_dates=['pickup_datetime'])


# **II Data Exploration**

# In[66]:


df_train.head()


# In[67]:


df_train.describe()


# In[68]:


df_test.head()


# In[69]:


df_test.describe()


# **III Data Preprocessing**

# In[70]:


plt.subplots(figsize=(18,6))
plt.title("results")
df_train.boxplot();


# In[71]:


df_train = df_train[df_train['passenger_count']>=1]
df_train = df_train[df_train['trip_duration']<=5000]

df_train['year'] = df_train['pickup_datetime'].dt.year
df_train['month'] = df_train['pickup_datetime'].dt.month
df_train['day'] = df_train['pickup_datetime'].dt.day
df_train['hour'] = df_train['pickup_datetime'].dt.hour
df_train['minute'] = df_train['pickup_datetime'].dt.minute
df_train['second'] = df_train['pickup_datetime'].dt.second

df_train.info()


# In[72]:


selected_columns = ['year','month','day','hour','minute','second','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']

x_train = df_train[selected_columns]
y_train = df_train['trip_duration']
x_train.shape, y_train.shape


# In[73]:


rf = RandomForestRegressor()
random_split = ShuffleSplit(n_splits = 3, test_size = 0.05, train_size=0.1, random_state=0)
looses = -cross_val_score(rf, x_train, y_train, cv = random_split, scoring = 'neg_mean_squared_log_error')
looses = [np.sqrt(l) for l in looses]
np.mean(looses)


# In[74]:


rf.fit(x_train, y_train)


# In[75]:




df_test['year'] = df_test['pickup_datetime'].dt.year
df_test['month'] = df_test['pickup_datetime'].dt.month
df_test['day'] = df_test['pickup_datetime'].dt.day
df_test['hour'] = df_test['pickup_datetime'].dt.hour
df_test['minute'] = df_test['pickup_datetime'].dt.minute
df_test['second'] = df_test['pickup_datetime'].dt.second

x_test = df_test[selected_columns]


# In[76]:


pred_test = rf.predict(x_test)
pred_test.mean()


# In[77]:


submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[78]:


submission['trip_duration'] = pred_test


# In[79]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




