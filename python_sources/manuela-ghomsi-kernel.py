#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING LIBRARIES ##

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns
import pathlib as Path
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## LOADING THE TRAIN DATA

# In[23]:


df = pd.read_csv('../input/train.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
df.head()


# In[24]:


df.info()


# In[25]:


df = df[df.passenger_count >= 1]
df = df[df.trip_duration >= 120]
df = df[df.trip_duration <= 7200]


# In[26]:


sns.distplot(df['trip_duration'], label='trip duration')
plt.legend();


# ## Creating addotional columns from pickup datetime

# In[27]:


df['pickup_year'] = df['pickup_datetime'].dt.year
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
df['hour'] = df['pickup_datetime'].dt.hour
df.head()


# ## Creating addotional columns from dropoff datetime

# In[28]:


df.describe()


# In[29]:


selected_columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
                    'dropoff_latitude', 'pickup_year',
                    'pickup_month', 'pickup_day', 'hour', 'pickup_weekday']
X = df[selected_columns]
y = df['trip_duration']
X.shape, y.shape


# ## Shuffleling the data 

# In[30]:


cv = ShuffleSplit(1, test_size=0.01, train_size=0.02, random_state=0)


# ## Categorising hours to avoid numerics

# ## Training the data

# In[31]:


from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor()
losses = cross_val_score(rf, X, y, cv=cv, scoring='neg_mean_squared_log_error')
np.sqrt(- losses.mean())


# In[32]:


rf.fit(X, y)


# ## Loading the test file for prediction

# In[33]:


df_test = pd.read_csv('../input/test.csv')
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_test.head()


# In[34]:


df_test.info()


# In[35]:


df_test['pickup_year'] = df_test['pickup_datetime'].dt.year
df_test['pickup_month'] = df_test['pickup_datetime'].dt.month
df_test['pickup_day'] = df_test['pickup_datetime'].dt.day
df_test['pickup_weekday'] = df_test['pickup_datetime'].dt.weekday
df_test['hour'] = df_test['pickup_datetime'].dt.hour
df_test.head()


# In[36]:


X_test = df_test[selected_columns]


# In[37]:


y_pred = rf.predict(X_test)
y_pred.mean()


# ## LOADING THE SUBMISSION TIME

# In[38]:


submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[39]:


submission['trip_duration'] = y_pred
submission.head()


# In[40]:


submission.describe()


# In[41]:


submission.to_csv('submission.csv', index=False)


# In[42]:


get_ipython().system('ls')


# In[ ]:




