#!/usr/bin/env python
# coding: utf-8

# # import

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob, re
from sklearn import *
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
from math import ceil
import time
import sys
import gc
import pickle
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import jpholiday
import xgboost as xgb
import matplotlib
from matplotlib import pylab as plt
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.font_manager._rebuild()
plt.rcParams["font.family"] = "IPAexGothic"
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()
from fbprophet import Prophet
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from datetime import datetime
import datetime as dt
import datetime
import calendar
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
from datetime import date, timedelta
from pandas.plotting import scatter_matrix


# # Extract the data

# In[ ]:


items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')
cats = pd.read_csv('item_categories.csv')
train = pd.read_csv('sales_train.csv',parse_dates=["date"])
test  = pd.read_csv('test.csv').set_index('ID')


# In[ ]:


#refuce memory
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    return df


# In[ ]:


items.head()
del items['item_name']
gc.collect()
train = pd.merge(train, items, on= 'item_id', how='left')
test = pd.merge(test, items, on= 'item_id', how='left')
reduce_mem_usage(train)
reduce_mem_usage(test)


# In[ ]:


#data details
train.describe()


# In[ ]:


#unique number
print('Unique shop_id number is', train.shop_id.nunique())
print('Unique item_id number is', train.item_id.nunique())
print('Unique item_category_id number is', train.item_category_id.nunique())


# In[ ]:


# add time features
train['date_year'] = train['date'].dt.year
train['date_quarter'] = train['date'].dt.quarter
train['week_count'] = train['date'].dt.week
train['date_month'] = train['date'].dt.month
train['date_day'] = train['date'].dt.day
train['date_dow'] = train['date'].dt.weekday


# In[ ]:


#item_id per item_category_id
x=train.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.reset_index()
plt.figure(figsize=(20,8))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("item_id per item_category_id")
plt.ylabel('# of item_id', fontsize=12)
plt.xlabel('item_category_id', fontsize=12)


# some item_categories have many items.  top #17 
# we can figure out what category is
# 

# In[ ]:


#ishop_id per item_category_id
x=train.groupby(['item_category_id']).count()
x=x.sort_values(by='shop_id',ascending=False)
x=x.reset_index()
plt.figure(figsize=(20,8))
ax= sns.barplot(x.item_category_id, x.shop_id, alpha=0.8)
plt.title("shop_id per item_category_id")
plt.ylabel('# of shop_id', fontsize=12)
plt.xlabel('item_category_id', fontsize=12)


# some item_categories have many shops. top #17 we can figure out what category is
# 
# 

# In[ ]:


#box plot 
plt.rcParams["font.family"] = "IPAexGothic"

var = 'item_category_id'
data = pd.concat([train['item_cnt_day'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(50, 10))
fig = sns.boxplot(x=var, y="item_cnt_day", data=data)
fig.axis(ymin=0, ymax=1000);


# let's zoom in this details 

# In[ ]:


#box plot 
plt.rcParams["font.family"] = "IPAexGothic"

var = 'item_category_id'
data = pd.concat([train['item_cnt_day'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(40, 5))
fig = sns.boxplot(x=var, y="item_cnt_day", data=data)
fig.axis(ymin=0, ymax=100);


# #8#9#71 category items are sold a lot

# In[ ]:


plt.rcParams["font.family"] = "IPAexGothic"
x=train.groupby(['date_year']).mean()
x=x.sort_values(by='item_cnt_day',ascending=False)
x=x.reset_index()
# #plot
plt.figure(figsize=(10,10))
ax= sns.barplot(x.date_year, x.item_cnt_day, alpha=0.8)
plt.title("item_cnt_day per date_year")
plt.ylabel('# of item_cnt_day', fontsize=12)
plt.xlabel('date_year', fontsize=12)

plt.show()


# It seems the average number of sales(item_cnt day) is same in year

# In[ ]:


plt.rcParams["font.family"] = "IPAexGothic"
x=train.groupby(['date_month']).mean()
x=x.sort_values(by='item_cnt_day',ascending=False)
x=x.reset_index()
# #plot
plt.figure(figsize=(10,10))
ax= sns.barplot(x.date_month, x.item_cnt_day, alpha=0.8)
plt.title("item_cnt_day per date_month")
plt.ylabel('# of item_cnt_day', fontsize=12)
plt.xlabel('date_month', fontsize=12)

plt.show()


# It seems the average number of sales(item_cnt day) is different is month

# In[ ]:


plt.rcParams["font.family"] = "IPAexGothic"
x=train.groupby(['date_day']).mean()
x=x.sort_values(by='item_cnt_day',ascending=False)
x=x.reset_index()
# #plot
plt.figure(figsize=(10,10))
ax= sns.barplot(x.date_day, x.item_cnt_day, alpha=0.8)
plt.title("item_cnt_day per date_day")
plt.ylabel('# of item_cnt_day', fontsize=12)
plt.xlabel('date_day', fontsize=12)

plt.show()


# It seems the average number of sales(item_cnt day) is different in a day

# In[ ]:


####In fact I usually do this after I create more features#####
#features correlation
corrmat =train.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.00)
sns.heatmap(corrmat, vmax=.8, square=True,annot=True, fmt= '.2f')


# In[ ]:


####In fact I usually do this after I create more features#####
cols = ['item_cnt_day', 'item_id', 'item_price','shop_id'] 
sns.set(font_scale=1.25)
sns.pairplot(train[cols], size=3)
plt.show();


# In[ ]:


to be continue

