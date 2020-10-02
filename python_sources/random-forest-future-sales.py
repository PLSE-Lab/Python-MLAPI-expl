#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime


# In[ ]:


train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")


# In[ ]:


test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")


# In[ ]:


submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
item_cats = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
train.describe()


# In[ ]:


display(train.head())
print(train.shape)


# In[ ]:


train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year


# In[ ]:


train = train.drop(['date', 'item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})

train.head()


# In[ ]:


shop_item_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})
train.head()


# In[ ]:


shop_prev_month = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_mean'})
shop_prev_month.head()


# In[ ]:


train = pd.merge(train, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.)
train = pd.merge(train, items, how='left', on='item_id')
train = pd.merge(train, item_cats, how='left', on='item_category_id')
train = pd.merge(train, shops, how='left', on='shop_id')

train.head()


# In[ ]:


test['month'] = 11
test['year'] = 2015
test['date_block_num']=34


# In[ ]:


test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.)
test = pd.merge(test, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.)
test = pd.merge(test, items, how='left', on='item_id')
test = pd.merge(test, item_cats, how='left', on='item_category_id')
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.
test.head()


# In[ ]:


for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[c].unique())+list(test[c].unique()))
    train[c] = lbl.transform(train[c].astype(str))
    test[c] = lbl.transform(test[c].astype(str))


# In[ ]:


col = [c for c in train.columns if c not in ['item_cnt_month']]
x1 = train[train['date_block_num']<33]
y1 = np.log1p(x1['item_cnt_month'].clip(0., 20.))
x1=x1[col]
x2 = train[train['date_block_num'] == 33]
y2 = np.log1p(x2['item_cnt_month'].clip(0., 20.))
x2=x2[col]


# In[ ]:


train.head()


# In[ ]:


reg = ensemble.ExtraTreesRegressor(n_estimators=40, n_jobs=-1, max_depth=15, random_state=18)
reg.fit(x1, y1)
print('RMSE value is: ', np.sqrt(metrics.mean_squared_error(y2.clip(0., 20.), reg.predict(x2).clip(0., 20.))))


# In[ ]:


reg.fit(train[col], train['item_cnt_month'].clip(0., 20.))


# In[ ]:


test.head()


# In[ ]:


test['item_cnt_month'] = reg.predict(test[col]).clip(0.,20.)
test[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)


# In[ ]:


test['item_cnt_month'] = np.expm1(test['item_cnt_month'])
test[['ID', 'item_cnt_month']].to_csv('final_submission.csv', index=False)


# In[ ]:


# DISCLAIMER: THIS CODE IS A RESULT OF FOLLOWING THIS TUTORIAL: https://www.youtube.com/watch?v=PD5QguzEdNs

