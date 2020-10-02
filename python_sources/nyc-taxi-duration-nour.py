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
from pathlib import Path
import matplotlib.pyplot as plt 
import seaborn as sns
 
from sklearn.linear_model import LinearRegression, Lasso
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('head -n ../input/sample_submission.csv')


# # Chargement de la Data & Analyse exploiratoire
# 

# In[ ]:


train = pd.read_csv("../input/train.csv",index_col=0)
test = pd.read_csv("../input/test.csv",index_col=0)


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


len(train[train['store_and_fwd_flag'] == 'N']) == len(train['store_and_fwd_flag'])


# In[ ]:


train['flag'] = np.where(train['store_and_fwd_flag']=='N', 0, 1)


# In[ ]:


test['flag'] = np.where(test['store_and_fwd_flag']=='N', 0, 1)


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime) 


# In[ ]:


test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime) 


# In[ ]:


train['pickup_date'] = train['pickup_datetime'].dt.date 
train['pickup_time'] = train['pickup_datetime'].dt.time 
train['pickup_month'] = train['pickup_datetime'].dt.month 
train['pickup_hour'] = train['pickup_datetime'].dt.hour 
train['pickup_weekday'] = train['pickup_datetime'].dt.dayofweek
 
test['pickup_month'] = test['pickup_datetime'].dt.month 
test['pickup_hour'] = test['pickup_datetime'].dt.hour 
test['pickup_weekday'] = test['pickup_datetime'].dt.dayofweek


# ## Extraction de la distance

# In[ ]:


train['dist_long'] = train['pickup_longitude'] - train['dropoff_longitude']
test['dist_long'] = test['pickup_longitude'] - test['dropoff_longitude']

train['dist_lat'] = train['pickup_latitude'] - train['dropoff_latitude']
test['dist_lat'] = test['pickup_latitude'] - test['dropoff_latitude']

train['dist'] = np.sqrt(np.square(train['dist_long']) + np.square(train['dist_lat']))
test['dist'] = np.sqrt(np.square(test['dist_long']) + np.square(test['dist_lat']))


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


train.drop(['vendor_id'], axis = 1, inplace=True)
train.drop(['store_and_fwd_flag'], axis = 1, inplace=True)
train.drop(['pickup_datetime'], axis = 1, inplace=True)
train.drop(['dropoff_datetime'], axis = 1, inplace=True)
train.drop(['pickup_date'], axis = 1, inplace=True)
train.drop(['pickup_time'], axis = 1, inplace=True)



test.drop(['vendor_id'], axis = 1, inplace=True)
test.drop(['store_and_fwd_flag'], axis = 1, inplace = True)
test.drop(['pickup_datetime'], axis = 1, inplace = True)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:


X = train
X.head()


# In[ ]:


y = train['trip_duration']
y.head() ;


# In[ ]:


X = train.drop(columns='trip_duration')
X.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=25, random_state=42, n_jobs=-1)


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
Loss = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_squared_log_error")


# In[ ]:


RMSLE = np.sqrt(- Loss)
RMSLE


# In[ ]:


rf.fit(X,y)


# In[ ]:


pred = rf.predict(test)


# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
arr_id = submit['id']
C1 = pd.DataFrame({'id': arr_id, 'trip_duration': pred})
C1.head()


# In[ ]:


C1.to_csv("to-submit.csv", index=False)

