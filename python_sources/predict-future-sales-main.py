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


from sklearn import *
import nltk, datetime

train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
items_cats = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

Prepare the data time
# In[ ]:


train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date', 'item_price'], axis=1)


# get all columns except for 'item_cnt_day', reset the index (ie. the groupby column does not become an index) and then sum the item_cnt_day 
# rename the item_cnt_day column appropriately as it now contains the sum of item count per month.

# In[ ]:


train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day': 'item_cnt_month'})


# In[ ]:


train.head()


# ### Find the monthly mean

# In[ ]:


shop_item_mean = train[['shop_id', 'item_id', 'item_cnt_month']].groupby(['shop_id', 'item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean = shop_item_mean.rename(columns={'item_cnt_month': 'item_cnt_month_mean'})


# In[ ]:


shop_item_mean.head()


# > Add the mean feature to the train set

# In[ ]:


train =  pd.merge(train, shop_item_mean, how='left',on=['shop_id', 'item_id'])


# Add Last Month

# In[ ]:


shop_prev_month = train[train['date_block_num']==33 ][['shop_id', 'item_id', 'item_cnt_month']]
shop_prev_month = shop_prev_month.rename(columns={'item_cnt_month': 'item_cnt_prev_month'})


# In[ ]:


shop_prev_month.head()


# Add previous month feature to dataset

# In[ ]:


train = pd.merge(train, shop_prev_month, how='left', on=['shop_id', 'item_id']).fillna(0.)


# In[ ]:


train.head()


# Add all item features

# In[ ]:


train = pd.merge(train, items, how='left', on='item_id') 


# Adding item category features 

# In[ ]:


train = pd.merge(train, items_cats, how='left', on='item_category_id')


# Adding Shop features

# In[ ]:


train = pd.merge(train, shops, how="left", on="shop_id")


# In[ ]:


train.head()


# #### Test Dataset
# Adding November 2015

# In[ ]:


test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34


#   Add  Mean feature

# In[ ]:


test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id'])


# Add Previous month feature

# In[ ]:


test = pd.merge(test, shop_prev_month, how="left", on=['shop_id', 'item_id']).fillna(0.)


# Add all items features

# In[ ]:


test = pd.merge(test, items, how='left', on=['item_id'])


# Add item category features

# In[ ]:


test = pd.merge(test, items_cats, how='left', on=['item_category_id'])


# Add shop category features

# In[ ]:


test = pd.merge(test, shops, how='left', on=['shop_id'])


# create a new column for predicting the item count per month

# In[ ]:


test['item_cnt_month']=0


# In[ ]:


test.head()


# Label Encoding

# In[ ]:


for c in ['shop_name', 'item_name', 'item_category_name']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[c].unique())+ list(test[c].unique()))
    train[c] = lbl.transform(train[c].astype(str))
    test[c] = lbl.transform(test[c].astype(str))
    print(c)


# Train and predict using Random forest algorithm

# In[ ]:


col = [c for c in train.columns if c not in ['item_cnt_month']]
x1 = train[train['date_block_num']<33]
y1 = np.log1p(x1['item_cnt_month'].clip(0.,20.))
x1 = x1[col]
x2 = train[train['date_block_num']==33]
y2 = np.log1p(x2['item_cnt_month'].clip(0.,20.))
x2 = x2[col]

reg = ensemble.ExtraTreesRegressor(n_estimators=25, n_jobs=1, max_depth=20, random_state=18)
reg.fit(x1,y1)
print('RMSE value is :', np.sqrt(metrics.mean_squared_error(y2.clip(0.,20.),reg.predict(x2.clip(0.,20.)))) )


# In[ ]:


reg.fit(train[col], train['item_cnt_month'].clip(0.,20.))
test['item_cnt_month'] = reg.predict(test[col]).clip(0.,20.)
test[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)

test['item_cnt_month'] = np.expm1(test['item_cnt_month'])
test[['ID','item_cnt_month']].to_csv('final_submission.csv', index=False)


# In[ ]:


test


# In[ ]:


test[test['ID']==58]


# In[ ]:




