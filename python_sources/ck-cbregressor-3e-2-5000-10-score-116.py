#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
import math
from matplotlib import pyplot as plt
import lightgbm
from lightgbm import *
import catboost

from sklearn.ensemble import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.metrics import *
from sklearn.preprocessing import *

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

import os


# In[ ]:


def process_date(x, start_time=[]):
    x = x.copy()
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in x['date']]
    SECONDS_IN_DAY = 24 * 60 * 60
    if not start_time:
        start_time.append(min(dates).timestamp() / SECONDS_IN_DAY)
    start_time = start_time[0]
    x['timestamp'] = [x.timestamp() / SECONDS_IN_DAY - start_time for x in dates]
    x['year'] = [x.year for x in dates]
    x['month'] = [x.month for x in dates]
    x['day'] = [x.day for x in dates]
    x['weekday'] = [x.weekday() for x in dates]
    x.drop('date', axis=1, inplace=True)
    return x


# In[ ]:


def process(x, y=None, *, debug=True):
    x = process_date(x)
    x.drop(['id', 'year', 'day', 'weekday', 'timestamp'], axis=1, inplace=True)
    
    #mask = x['renovation_year'] == 0
    #x.loc[mask, 'renovation_year'] = x['build_year'][mask]
    
    # OneHotEncoding of Month
    data = np.zeros((x.shape[0], 12), dtype=np.int)
    data[range(x.shape[0]), x['month'] - 1] = 1
    x = x.join(pd.DataFrame(data, index=x.index), rsuffix="month")
    x.drop('month', axis=1, inplace=True)
    
    if y is not None:
        # Removing outliers
        y = y.copy()
        mask = y['price'] > 4000
        print(f"Deleted {mask.sum()} elements") if debug else None
        x.drop(y.index[mask], inplace=True)
        y.drop(y.index[mask], inplace=True)
        y = np.array(y).reshape(y.shape[0])
        return x, y
    return x


# In[ ]:


def cross_val(model, x_train, y_train, test_percs=[0.3, 0.25, 0.2, 0.15, 0.1, 0.05], *, debug=True):
    sum_rmse = 0
    for test_perc in test_percs:
        test_size = round(x_train.shape[0] * test_perc)
        cur_x_train, cur_x_test = x_train[:-test_size], x_train[-test_size:]
        cur_y_train, cur_y_test = y_train[:-test_size], y_train[-test_size:]
        
        cur_x_train, cur_y_train = process(cur_x_train, cur_y_train, debug=False)
        cur_x_test = process(cur_x_test)
        
        model.fit(cur_x_train, cur_y_train, eval_set=(cur_x_test, cur_y_test))
        cur_y_pred = model.predict(cur_x_test)
        cur_rmse = rmse(cur_y_test, cur_y_pred)
        
        print(f"Test%: {test_perc} RMSE: {cur_rmse}") if debug else None
        sum_rmse += cur_rmse
    return sum_rmse / len(test_percs)


# In[ ]:


def super_cross_val(model, x_train, y_train, percs=[i / 10 for i in range(5, 11)], *, debug=True):
    sum_rmse = 0
    for perc in percs:
        full_size = round(x_train.shape[0] * perc)
        mean_rmse = cross_val(model, x_train.iloc[:full_size], y_train.iloc[:full_size])
        print(f"Current%: {perc} Size: {full_size} RMSE: {mean_rmse}") if debug else None
        sum_rmse += mean_rmse
    return sum_rmse / len(percs)


# # Test

# In[ ]:


x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')


# In[ ]:


model_class = catboost.CatBoostRegressor
kwargs = {
    'learning_rate': 3e-2,
    'n_estimators': 5000,
    'max_depth': 10
}


# In[ ]:


# In the name of LGBMRegressor(learning_rate=1e-2, n_estimators=2000)
model = model_class(**kwargs)
#cross_val(model, x_train, y_train)


# # Master

# In[ ]:


x_train = pd.read_csv('../input/train_data.csv', index_col='index')
y_train = pd.read_csv('../input/train_target.csv', index_col='index')
x_test = pd.read_csv('../input/test_data.csv', index_col='index')
idx = x_test.index.values


# In[ ]:


x_train, y_train = process(x_train, y_train)
x_test = process(x_test)


# In[ ]:


model = model_class(**kwargs).fit(x_train, y_train)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


my_submission = pd.DataFrame({'index': idx, 'price': y_pred})
my_submission.to_csv(f'./{model.__class__.__name__}{learning_rate}_{n_estimators}_NoOutliers_OHCMonth.csv', index=False)

