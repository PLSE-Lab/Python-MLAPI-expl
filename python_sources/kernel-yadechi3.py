#!/usr/bin/env python
# coding: utf-8

# In[132]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import seaborn as sns
import pathlib as Path
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split, ShuffleSplit


# In[133]:


df = pd.read_csv('../input/train.csv')


# In[134]:


def split_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    df['year_' + column_name] = df[column_name].dt.year
    df['month_' + column_name] = df[column_name].dt.month
    df['day_' + column_name] = df[column_name].dt.day
    df['weekday_' + column_name] = df[column_name].dt.weekday + 1
    df['hour_' + column_name] = df[column_name].dt.hour
    df['minute_' + column_name] = df[column_name].dt.minute
    return df


# In[135]:


df.head()


# # PREPROCESSING

# ### Splitting a datetime

# In[136]:


new_df = split_datetime(df, 'pickup_datetime')
new_df.shape


# ### Column 'store_and_fwd_flag' into binary values

# In[137]:


# new_df['store_and_fwd_flag'] = pd.get_dummies(new_df['store_and_fwd_flag'])
# new_df.head()


# ### Applying some filters

# #### 1st filter

# In[138]:


new_df = new_df[new_df['passenger_count'] != 0]
new_df.shape


# #### 2nd filter

# In[139]:


new_df = new_df[new_df['trip_duration'] <= 1800]
new_df.shape


# In[140]:


new_df = new_df[new_df['trip_duration'] >= 360]
new_df.shape


# ### Selecting columns to use

# In[141]:


selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude', 'month_pickup_datetime', 'hour_pickup_datetime']


# ## Defining target & features

# In[142]:


X_full = new_df[selected_columns]
y_full = new_df['trip_duration']
X_full.shape, y_full.shape


# ### Splitting my dataset

# In[143]:


X_train, X_valid, y_train, y_valid = train_test_split(
            X_full, y_full, test_size=0.33, random_state=42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# In[144]:


#X_train, X_valid, y_train, y_valid = train_test_split(
#            X_train_used, y_train_used, test_size=0.33, random_state=50)
#X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# # Training model

# In[145]:


#rf = RandomForestRegressor()


# In[146]:


xbr = XGBRegressor(n_jobs=-1)


# In[147]:


#params_grid = {
#    'colsample_bytree':[0.4,0.6,0.8],
#    'gamma':[0.03,0.3],
#    'min_child_weight':[1.5,6],
#    'learning_rate':[0.1,0.07],
#    'reg_alpha':[1e-5, 1e-2,  0.75],
#    'reg_lambda':[1e-5, 1e-2, 0.45],
#    'subsample':[0.6,0.95],
#    'max_depth': [3, 5],
    #'min_samples_leaf': [1, 3, 8, 12]
#}


# In[148]:


# kf = KFold(n_splits=5, random_state=1) 


# In[149]:


# gsc = GridSearchCV(xbr, params_grid, n_jobs=-1, verbose=10, scoring='neg_mean_squared_log_error')


# In[150]:


# gsc.fit(X_train, y_train)


# In[151]:


# gsc.estimator


# In[152]:


# gsc.best_index_


# In[153]:


# gsc.best_score_


# In[154]:


# gsc.best_params_


# In[155]:


# gsc.n_splits_


# In[156]:


#cv = ShuffleSplit(n_splits=5, train_size=0.75, random_state=0)


# In[157]:


# rf_v2 = RandomForestRegressor(max_depth=15, min_samples_leaf=12)


# In[158]:


xbr.fit(X_train, y_train)


# In[159]:


xbr.feature_importances_


# In[161]:


#losses = -cross_val_score(rf_v2, X_train_used, y_train_used, cv=gsc.best_index_, scoring='neg_mean_squared_log_error')
#losses.mean()


# Real score value

# In[162]:


#losses = [np.sqrt(l) for l in losses]
#np.mean(losses)


# In[163]:


#rf_v2.fit(X_train_used, y_train_used)


# In[164]:


#rf_v2.feature_importances_


# In[165]:


y_pred_valid = xbr.predict(X_valid)


# In[166]:


#y_pred = rf_v2.predict(X_train_unused)


# In[167]:


y_pred_valid.mean()


# In[168]:


np.mean(y_valid)


# In[169]:


df_test = pd.read_csv('../input/test.csv')


# In[170]:


df_test.head()


# In[171]:


df_test = split_datetime(df_test, 'pickup_datetime')


# In[172]:


# df_test['store_and_fwd_flag'] = pd.get_dummies(df_test['store_and_fwd_flag'])


# In[173]:


X_test = df_test[selected_columns]


# In[174]:


y_pred_test = xbr.predict(X_test)


# In[175]:


y_pred_test.mean()


# In[176]:


submission = pd.read_csv('../input/sample_submission.csv') 
submission.head()


# In[177]:


submission['trip_duration'] = y_pred_test


# In[178]:


submission.describe()


# In[179]:


submission.to_csv('submission.csv', index=False)


# In[180]:


get_ipython().system('ls')


# In[ ]:




