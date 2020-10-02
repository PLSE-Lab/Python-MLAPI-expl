#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[27]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[28]:


df_train.head()


# In[29]:


df_train.info()


# In[30]:


df_train.describe()


# **PLOT**

# In[31]:



plt.plot( df_train['trip_duration'], data=df_train, linestyle='none', marker='o')
plt.show()


# **FILTRE**

# In[38]:


df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
df_train['year'] = df_train['pickup_datetime'].dt.year
df_train['month'] = df_train['pickup_datetime'].dt.month
df_train['day'] = df_train['pickup_datetime'].dt.day
df_train['hour'] = df_train['pickup_datetime'].dt.hour
df_train['weekday'] = df_train['pickup_datetime'].dt.weekday
df_train['dayofyear'] = df_train['pickup_datetime'].dt.dayofyear

df_train = df_train[df_train['trip_duration']<= 50000]
df_train = df_train[df_train['passenger_count']>= 1]  
df_train.describe()


# In[39]:


columns= ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'year', 'month', 'day', 'hour', 'weekday', 
          'dayofyear']  


# In[40]:


df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_test['year'] = df_test['pickup_datetime'].dt.year
df_test['month'] = df_test['pickup_datetime'].dt.month
df_test['day'] = df_test['pickup_datetime'].dt.day
df_test['hour'] = df_test['pickup_datetime'].dt.hour
df_test['weekday'] = df_test['pickup_datetime'].dt.weekday
df_test['dayofyear'] = df_test['pickup_datetime'].dt.dayofyear
    
df_test.describe()


# In[41]:


X_train = df_train[columns]
y_train = df_train['trip_duration']


# In[42]:


rf = RandomForestRegressor()
cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.1, random_state=200)
losses = -cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_squared_log_error')
losses = [np.sqrt(l) for l in losses]
losses


# In[43]:


rf.fit(X_train, y_train)


# In[44]:


X_test = df_test[columns]
y_pred = rf.predict(X_test)
y_pred.mean()


# In[45]:


submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[46]:


len(y_pred)


# In[47]:


submission['trip_duration'] = y_pred


# In[48]:


submission.to_csv('submission.csv', index=False)


# In[49]:


get_ipython().system('ls')


# In[ ]:




