#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import pathlib as Path
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os
print(os.listdir("../input"))


# In[9]:


df = pd.read_csv("../input/train.csv")
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df[df['trip_duration'] <= 5000]
df.info()


# In[10]:


## split pickup_datetime
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_minute'] = df['pickup_datetime'].dt.minute
df['pickup_second'] = df['pickup_datetime'].dt.second


# In[11]:


df.head()


# In[12]:


## X creation
X = df[[
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude', 
    'pickup_hour',
    'pickup_minute',
    'pickup_second',
   ]]


# In[13]:


## Y creation
Y = df[['trip_duration']]

## Shape
X.shape, Y.shape


# In[14]:


## Cross validation
rf = RandomForestRegressor()
score = -cross_val_score(rf, X, Y, cv=5, scoring='neg_mean_squared_log_error')
score.mean()


# In[15]:


rf.fit(X, Y)


# In[16]:


## DF Test creation
df_test = pd.read_csv('../input/test.csv')
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_test.head()


# In[17]:


## split pickup_datetime
df_test['pickup_hour'] = df_test['pickup_datetime'].dt.hour
df_test['pickup_minute'] = df_test['pickup_datetime'].dt.minute
df_test['pickup_second'] = df_test['pickup_datetime'].dt.second


# In[20]:


## X creation
X_test = df_test[[
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude', 
    'pickup_hour',
    'pickup_minute',
    'pickup_second',
    ]]


# In[21]:


## y prediction and mean for this prediction
y_pred = rf.predict(X_test)
y_pred.mean()


# In[22]:


## submit creation
submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[23]:


## submission for the y prediction
submission['trip_duration'] = y_pred
submission.head()


# In[24]:


## describe submission
submission.describe()


# In[25]:


## send submission
submission.to_csv('submission.csv', index=False)


# In[26]:


get_ipython().system('ls')


# In[ ]:




