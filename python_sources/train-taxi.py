#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
#print(os.listdir("../input"))
from os import path
# Any results you write to the current directory are saved as output.


# ## Data loading

# In[3]:


# Load the data train
TRAIN_PATH = path.join('..', 'input','train.csv')
df = pd.read_csv(TRAIN_PATH)
df.head()


# In[4]:


# Load the data test
TRAIN_PATH = path.join('..', 'input','test.csv')
df_test = pd.read_csv(TRAIN_PATH)
df_test.head()


# In[5]:


# Load the data submission
SUBMIT_PATH = path.join('..', 'input','sample_submission.csv')
df_submission = pd.read_csv(SUBMIT_PATH)
df_submission.head()


# ## Data exploration

# ### Global informations

# In[6]:


# Show informations about all data
# df['store_and_fwd_flag'].nunique()
# df.shape
# df.describe()
df.info()


# ### Count of passengers

# In[7]:


plt.figure(figsize=(15, 4))
plt.hist(x='passenger_count', data=df, orientation='horizontal');


# We can see that between 1 and 6 there is no data. Indeed, there were courses which have been recorded with no passengers. Beside, a taximan can't take more than 4 passengers. We'll filter data passengers to get ride of that.

# ### Trip duration

# In[8]:


plt.figure(figsize=(20, 5))
df['trip_duration'].hist();


# We can see above a large amount of trips which lasted for days because they are over 24 hours (= 86400 seconds). In fact, we'll filter data to see trips less than 2 hours (= 7200 seconds).

# ## Data processing

# In[9]:


# change pickup_datetime to datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])


# In[10]:


# encode store_and_fwd_flag column and add column to see if it's the night or not
def preprocess(df):
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_min'] = df['pickup_datetime'].dt.minute
    df['store_and_fwd_flag_codes'] = df['store_and_fwd_flag'].astype('category').cat.codes
    df['is_night'] = (df['pickup_hour'] > 18) & (df['pickup_hour'] < 7)
    
preprocess(df)
preprocess(df_test)


# In[11]:


df.head()


# In[12]:


# filter DF by remove rows with 0 passengers and trip duration over 7200 secs
filter_ = (df['passenger_count'] > 0) & (df['trip_duration'] < 7200)
df = df[filter_]
df.shape


# ## Features engineering

# ### Config

# In[13]:


TARGET = 'trip_duration'
FEATURES = df.columns.drop(['trip_duration','pickup_datetime', 'dropoff_datetime', 'id', 'store_and_fwd_flag'])


# ### Split

# In[14]:


def split_dataset(df, features, target=TARGET):
    X = df[features]
    y = df[target]
    
    return X, y


# In[24]:


X_train, y_train = split_dataset(df, features=FEATURES)
X_train.shape, y_train.shape


# ## Modeling

# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

rf = RandomForestRegressor()
kf = KFold(n_splits=5, random_state=1)


# ### Cross validation

# In[17]:


rf.fit(X_train, y_train)


# In[18]:


losses = cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error')
losses = [np.sqrt(-l) for l in losses]
np.mean(losses)


# ## Prediction

# In[19]:


# Re-instantiate RF for test and fit
rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# In[20]:


X_test = df_test[FEATURES]


# In[21]:


# test prediction
y_test_pred = rf.predict(X_test)
y_test_pred.mean()


# ## Submission

# In[22]:


df_submission['trip_duration'] = y_test_pred
df_submission.head()


# In[23]:


# comparison
train_mean = df['trip_duration'].mean()
pred_mean = df_submission['trip_duration'].mean()
train_mean, pred_mean


# In[ ]:


df_submission.to_csv('submission.csv', index=False)

