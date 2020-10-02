#!/usr/bin/env python
# coding: utf-8

# In[40]:


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

# Any results you write to the current directory are saved as output.


# In[41]:


df = pd.read_csv('../input/train.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.info()


# In[42]:


df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday
df.head()


# In[43]:


df.describe()


# In[44]:


selected_columns = ['season', 'holiday', 'workingday', 'weather', 
                    'temp', 'atemp', 'humidity', 'windspeed', 'year',
                    'month', 'day', 'hour', 'weekday']
X = df[selected_columns]
y = df['count']
X.shape, y.shape


# In[45]:


rf = RandomForestRegressor()
losses = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_log_error')
losses.mean()


# In[46]:


rf.fit(X, y)


# In[47]:


df_test = pd.read_csv('../input/test.csv')
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['day'] = df_test['datetime'].dt.day
df_test['hour'] = df_test['datetime'].dt.hour
df_test['weekday'] = df_test['datetime'].dt.weekday
df_test.head()


# In[48]:


X_test = df_test[selected_columns]


# In[49]:


y_pred = rf.predict(X_test)
y_pred.mean()


# In[50]:


submission = pd.read_csv('../input/sampleSubmission.csv') 
submission.head()


# In[51]:


submission['count'] = y_pred
submission.head()


# In[52]:


submission.describe()


# In[53]:


submission.to_csv('submission.csv', index=False)


# In[54]:


get_ipython().system('ls')


# In[ ]:




