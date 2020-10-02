#!/usr/bin/env python
# coding: utf-8

# In[258]:


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
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, ShuffleSplit


# In[259]:


df = pd.read_csv('../input/train.csv', index_col='id')


# In[260]:


def split_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    df['year_' + column_name] = df[column_name].dt.year
    df['month_' + column_name] = df[column_name].dt.month
    df['day_' + column_name] = df[column_name].dt.day
    df['weekday_' + column_name] = df[column_name].dt.weekday
    df['hour_' + column_name] = df[column_name].dt.hour
    df['minute_' + column_name] = df[column_name].dt.minute
    return df


# In[261]:


df.head()


# **PREPROCESSING**

# Splitting a datetime

# In[262]:


new_df = split_datetime(df, 'pickup_datetime')
new_df.shape


# Applying some filters

# 1st filter

# In[263]:


new_df = new_df[new_df['passenger_count'] >= 1]
new_df.shape


# 2nd filter

# In[264]:


new_df = new_df[new_df['trip_duration'] <= 7200]
new_df.shape


# In[265]:


new_df = new_df[new_df['trip_duration'] >= 300]
new_df.shape


# Selecting columns to use

# In[266]:


selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                   'dropoff_latitude', 'day_pickup_datetime',
                   'hour_pickup_datetime', 'minute_pickup_datetime']


# **Defining target & features**

# In[267]:


X_full = new_df[selected_columns]
y_full = new_df['trip_duration']
X_full.shape, y_full.shape


# Splitting my dataset

# In[268]:


X_train_used, X_train_unused, y_train_used, y_train_unused = train_test_split(
            X_full, y_full, test_size=0.60, random_state=50)
X_train_used.shape, X_train_unused.shape, y_train_used.shape, y_train_unused.shape


# In[269]:


X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_used, y_train_used, test_size=0.33, random_state=50)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# Creating a RandomForestRegressor

# In[270]:


rf = RandomForestRegressor()


# In[271]:


params_grid = {
    'max_depth': [1, 3, 5, 10, 15],
    'min_samples_leaf': [1, 3, 8, 12]
}


# In[272]:


# kf = KFold(n_splits=5, random_state=1)


# In[273]:


# gsc = GridSearchCV(rf, params_grid, n_jobs=-1, cv=kf, verbose=3, scoring='neg_mean_squared_log_error')#


# In[274]:


# gsc.fit(X_train, y_train)


# In[275]:


# gsc.best_estimator_


# In[276]:


# gsc.best_index_


# In[277]:


cv = ShuffleSplit(1, test_size=0.01, train_size=0.5, random_state=0)


# Finding validation score

# In[278]:


losses = -cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
losses.mean()


# Real score value

# In[279]:


losses = [np.sqrt(l) for l in losses]
np.mean(losses)


# In[280]:


rf.fit(X_train, y_train)


# In[281]:


rf.feature_importances_


# In[282]:


y_pred = rf.predict(X_valid)


# In[283]:


y_pred.mean()


# In[284]:


np.mean(y_valid)


# In[285]:


df_test = pd.read_csv('../input/test.csv', index_col='id')


# In[286]:


df_test.head()


# In[287]:


df_test = split_datetime(df_test, 'pickup_datetime')


# In[288]:


X_test = df_test[selected_columns]


# In[289]:


y_pred_test = rf.predict(X_test)


# In[290]:


y_pred_test.mean()


# In[291]:


submission = pd.read_csv('../input/sample_submission.csv', index_col='id') 
submission.head()


# In[292]:


submission['trip_duration'] = y_pred_test


# In[293]:


submission.describe()


# In[294]:


submission.to_csv('submission.csv', index=False)


# In[295]:


get_ipython().system('ls')


# In[ ]:




