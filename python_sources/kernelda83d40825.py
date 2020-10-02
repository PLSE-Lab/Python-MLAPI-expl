#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt

train = pd.read_csv('../input/sales_train.csv')
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
train.head()


# In[ ]:


print("Train data shape: ", train.shape)
print("Test data shape: ", test.shape)


# In[ ]:


# remove noisy data
train = train[train['item_cnt_day']>=0]
train = train[train['item_cnt_day']<20]
print(train.item_cnt_day.describe())


# In[ ]:


# transform datetime
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

# drop unnecessary columns
train = train.drop(['date', 'item_price'], axis=1)

train.head()


# In[ ]:


# add monthly count feature
train = train.groupby(['date_block_num', 'shop_id', 'item_id', 'month', 'year'], as_index=False)[['item_cnt_day']].sum()
train.columns = ['date_block_num', 'shop_id', 'item_id', 'month', 'year', 'item_cnt_month']
train.head()


# In[ ]:


# calculate avarage sales of items by shop
shop_item_mean = train[['item_id', 'shop_id', 'item_cnt_month']].groupby(['item_id', 'shop_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_mean.columns = ['item_id', 'shop_id', 'item_cnt_month_mean']
shop_item_mean.head()


# In[ ]:


# add avareage sales as feature to train
train = pd.merge(train, shop_item_mean, how='left', on=['shop_id', 'item_id'])
train.head()


# In[ ]:


# calculate October 2015 sales
last_month_sales = train[train['date_block_num'] == 33][['shop_id', 'item_id', 'item_cnt_month']]
last_month_sales.columns = ['shop_id', 'item_id', 'item_cnt_last_month']
last_month_sales.head()


# In[ ]:


# add October 2015 as feature to train
train = pd.merge(train, last_month_sales, how='left', on=['shop_id', 'item_id']).fillna(0.)
train.head()


# In[ ]:


# add item_category as a feature 
train = pd.merge(train, items, how='left', on=['item_id'])
train.head()


# In[ ]:


# remove unnecesary item_name column
train = train.drop('item_name', axis=1)
train.head()


# In[ ]:


# prepare test dataset
test['date_block_num']=34
test['year']=2015
test['month']=11
# add all feautures to test dataset
test = pd.merge(test, shop_item_mean, how='left', on=['shop_id', 'item_id']).fillna(0.)
test = pd.merge(test, last_month_sales, how='left', on=['shop_id', 'item_id']).fillna(0.)
test = pd.merge(test, items, how='left', on=['item_id'])
test = test.drop('item_name', axis=1)
test['item_cnt_month']=0.
test.head()


# In[ ]:


train.corr()


# In[ ]:


# predict from past data
feature_cols = [c for c in train.columns if c not in ['item_cnt_month']]
X_train = train[train['date_block_num']<33]
y_train = X_train['item_cnt_month']

X_test = train[train['date_block_num']==33]
y_test = X_test['item_cnt_month']

X_train = X_train[feature_cols]
X_test = X_test[feature_cols]


# In[ ]:


# train model
regressor = ensemble.ExtraTreesRegressor(n_estimators=30, max_depth=15, n_jobs=-1, random_state=18)
model = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[ ]:


print("Accuracy is: ", model.score(X_test, y_test))


# In[ ]:


print("RMSE is: ", mean_squared_error(y_test, y_pred))


# In[ ]:


plt.scatter(y_pred, y_test, alpha=.75)
plt.xlabel('Predicted Count')
plt.ylabel('Actual Count')
plt.title('ExtraTreesRegressor Model')
plt.show()


# In[ ]:


# train all data and fill test columns
regressor.fit(train[feature_cols], train['item_cnt_month'])
test['item_cnt_month'] = regressor.predict(test[feature_cols])
test[['ID', 'item_cnt_month']].to_csv('submission.csv', index=False)

