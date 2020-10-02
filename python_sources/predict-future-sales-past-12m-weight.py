#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from datetime import date, datetime, timedelta
from time import time
from random import randint
from dateutil.relativedelta import relativedelta

import sys
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
sample = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# # Preprocessing
# 
# ## 1. Missing Values

# In[ ]:


msno.matrix(train)


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# No Missing Values! Lucky us :)

# ## 2. Dtypes and Formatting

# In[ ]:


train.dtypes


# In[ ]:


train.date[4]


# In[ ]:


def date_formatter(x):
    year = x[-4:]
    month = x[3:5]
    day = x[:2]
    stf = year + '-' + month + '-' + day
    return stf

train['date'] = train['date'].apply(date_formatter)
train['date'] = pd.to_datetime(train['date'])
train.head(3)


# # Exploratory Data Analysis
# ## 1. Trend, Seasonality

# In[ ]:


train.groupby('date_block_num')['item_cnt_day'].sum().plot(figsize=(16,9), color='royalblue', title="Monthly Sales Volume", linewidth=3)


# In[ ]:


train.groupby('shop_id')['item_cnt_day'].sum().plot(kind='bar', figsize=(16,9), color='royalblue', edgecolor='k', title="Sales Volume by shops")


# In[ ]:


train.groupby('date_block_num')['shop_id'].unique().apply(len).plot(figsize=(16,9), color='maroon', title="# shops each month")


# Findings So far:
# 1. Decreasing Trend
# 2. 12-month seasonality
# 3. Large Variation across shops
# 4. Number of shops on business vary time to time

# ## 2. Other Fact Checks

# In[ ]:


len(train['item_id'].unique())

# Many items on shelf


# In[ ]:


len(train['shop_id'].unique())


# In[ ]:


len(train['date_block_num'].unique())


# # Modeling
# ## 1. Statistics Approach

# In[ ]:


monthly = train.groupby('date_block_num')['item_cnt_day'].sum()

res = sm.tsa.seasonal_decompose(monthly, freq=12, model='multiplicative')
res.plot()
plt.show()


# In[ ]:


res = sm.tsa.seasonal_decompose(monthly, freq=12, model='additive')
res.plot()
plt.show()


# We can't tell just by looking whether multiplicative model assumption or additive assumption fits better with this dataset. We'll try them both

# In[ ]:


res = sm.tsa.seasonal_decompose(monthly, freq=12, model='multiplicative')
res.trend[-12:]


# In[ ]:


res.seasonal[-12:]


# In[ ]:


res.resid.mean()


# In[ ]:


trends = res.trend.dropna()
trends


# In[ ]:


deltas = [trends[i+1]-trends[i] for i in range(trends.index.min(), trends.index.max())]
deltas


# In[ ]:


e_delta = np.array(deltas).mean()
# 34th date_block_num is our target
e_trend = trends[trends.index.max()]+e_delta*7
e_season = res.seasonal[res.seasonal.index.max()-11]
e_value = e_trend*e_season
e_value


# This is our agg forecast assuming the multiplicative model. Let's try the additive

# In[ ]:


res = sm.tsa.seasonal_decompose(monthly, freq=12, model='additive')
res.resid.mean()


# In[ ]:


e_delta = np.array(deltas).mean()
# 34th date_block_num is our target
e_trend = trends[trends.index.max()]+e_delta*7
e_season = res.seasonal[res.seasonal.index.max()-11]
e_value = e_trend + e_season + res.resid.mean()
e_value


# It's only a slight difference, but it can be significant after all

# ## 2. Machine Learning Approach

# In[ ]:


from fbprophet import Prophet
model = Prophet(yearly_seasonality=True)
df = pd.DataFrame(monthly).reset_index().rename(columns={'date_block_num': 'ds', 'item_cnt_day': 'y'})
for i in range(df.ds.max()+1):
    in_date = date(2013, 1, 1) + relativedelta(months=i)
    df['ds'] = df['ds'].replace(i, in_date)
model.fit(df)
future = model.make_future_dataframe(periods=1, freq='MS')
forecast = model.predict(future)
forecast


# In[ ]:


e_value = forecast[forecast['ds']=='2015-11-01']['yhat']
e_value.values[0]


# This value lies somewhere inbetween the two values we got from multiplicative and additive seasonal decompose. So we could say all these results point at similar directions.

# # 4. Making Predictions for Submission

# In[ ]:


ct = train[train['shop_id'].isin(test['shop_id'])]
ct = ct[ct['date_block_num']>=22]
shop_ratio = ct.groupby('shop_id')['item_cnt_day'].sum()
item_ratio = ct.groupby(['shop_id', 'item_id'])['item_cnt_day'].sum()


# In[ ]:


agg = shop_ratio.sum()
shop = pd.DataFrame(shop_ratio)
shop['agg'] = np.ones(len(shop))*agg
shop['ratio'] = shop['item_cnt_day'] / shop['agg']
shop.head()


# In[ ]:


yhat = e_value.values[0]
shop['yhat'] = np.ones(len(shop))*yhat
shop['pred'] = shop['yhat']*shop['ratio']
shop.tail()


# In[ ]:


item = pd.DataFrame(item_ratio).reset_index()
item = item.merge(shop.reset_index()[['shop_id', 'item_cnt_day']].rename(columns={'item_cnt_day': 'agg'}), how='left', on='shop_id')
item.head()
# item = item.merge(shop.reset_index()[['shop_id', 'pred']], how='left', on='shop_id')


# In[ ]:


shop.head(3)


# In[ ]:


item['rate'] = item['item_cnt_day'] / item['agg']
item = item.merge(shop.reset_index()[['shop_id', 'pred']], how='left', on='shop_id')
item['cnt_pred'] = item['rate']*item['pred']
item.head()


# In[ ]:


sample.head(2)


# In[ ]:


test.head(2)


# In[ ]:


sub = test.merge(item[['shop_id', 'item_id', 'cnt_pred']], how='left', on=['shop_id', 'item_id']).rename(columns={'cnt_pred': 'item_cnt_month'})
sub.head()


# In[ ]:


submission = sub.fillna(0)[['ID', 'item_cnt_month']]
submission.to_csv('submission3.csv', index=False)


# In[ ]:


submission.head(5)


# In[ ]:




