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


path = '/kaggle/input/competitive-data-science-predict-future-sales/'


# In[ ]:


sales_train = pd.read_csv(path+'sales_train.csv')
sales_train.head(5)


# In[ ]:


item_categories = pd.read_csv(path+'item_categories.csv')
item_categories.head()


# In[ ]:


shops = pd.read_csv(path+'shops.csv')
shops.head()


# In[ ]:


items = pd.read_csv(path+'items.csv')
items.head()


# In[ ]:


test = pd.read_csv(path+'test.csv')
test.head()


# In[ ]:


test['date_block_num'] = 34


# In[ ]:


test = test.merge(items,how='left',on='item_id')


# In[ ]:


test.drop(['item_name'], axis=1,inplace= True)


# In[ ]:


test.head()


# In[ ]:


sales_train = sales_train.merge(shops,how='left',on='shop_id')


# In[ ]:


sales_train = sales_train.merge(items,how='left',on='item_id')


# In[ ]:


sales_train = sales_train.merge(item_categories,how='left',on='item_category_id')


# In[ ]:


sales_train.head()


# In[ ]:


sales_train['total_price_per_day'] = sales_train.item_cnt_day * sales_train.item_price


# In[ ]:


sales_train.head()


# In[ ]:


sales_train[['day','month','year']] = sales_train.date.str.split(".",expand=True)


# In[ ]:


sales_train.head()


# In[ ]:


sales_train_shop_item = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].sum()


# In[ ]:


sales_train_shop_item = sales_train_shop_item.reset_index()


# In[ ]:


sales_train_shop_item.rename(columns={"item_cnt_day":"item_cnt_month"}, inplace=True)
sales_train_shop_item.rename(columns={"total_price_per_day":"total_price_per_month"}, inplace=True)


# In[ ]:


sales_train_shop_item.head()


# In[ ]:


sales_min = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].min()


# In[ ]:


sales_min = sales_min.reset_index()


# In[ ]:


sales_min.head()


# In[ ]:


sales_train_shop_item['min_item_cnt_month'] = sales_min['item_cnt_day']
sales_train_shop_item['min_price_per_month'] = sales_min['total_price_per_day']


# In[ ]:


sales_train_shop_item.head()


# In[ ]:


sales_max = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].max()


# In[ ]:


sales_max = sales_max.reset_index()


# In[ ]:


sales_train_shop_item['max_item_cnt_month'] = sales_max['item_cnt_day']
sales_train_shop_item['max_price_per_month'] = sales_max['total_price_per_day']


# In[ ]:


sales_train_shop_item.head()


# In[ ]:


sales_avg = sales_train.groupby(by=['date_block_num','shop_id','item_id','item_category_id'])[['item_cnt_day','total_price_per_day']].mean()


# In[ ]:


sales_avg = sales_avg.reset_index()


# In[ ]:


sales_train_shop_item['avg_item_cnt_month'] = sales_avg['item_cnt_day']
sales_train_shop_item['avg_price_per_month'] = sales_avg['total_price_per_day']


# In[ ]:


test.head()


# In[ ]:


sales_train_shop_item.head()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# **For predicting total price**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]
y = sales_train_shop_item["total_price_per_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id']])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['total_price_per_month'] = y_test


# **For predicting min_item_count_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month"]]
y = sales_train_shop_item["min_item_cnt_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month']])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['min_item_cnt_month'] = y_test


# **For predicting min_price_per_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month"]]
y = sales_train_shop_item["min_price_per_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"min_item_cnt_month"]])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['min_price_per_month'] = y_test


# **For predicting max_item_cnt_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_price_per_month"]]
y = sales_train_shop_item["max_item_cnt_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month','min_price_per_month']])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['max_item_cnt_month'] = y_test


# **For predicting max_price_per_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","max_item_cnt_month"]]
y = sales_train_shop_item["max_price_per_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"max_item_cnt_month"]])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['max_price_per_month'] = y_test


# **For predicting avg_item_cnt_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","max_item_cnt_month","min_item_cnt_month"]]
y = sales_train_shop_item["avg_item_cnt_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month','max_item_cnt_month','min_item_cnt_month']])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['avg_item_cnt_month'] = y_test


# **For predicting avg_price_per_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","avg_item_cnt_month"]]
y = sales_train_shop_item["avg_price_per_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[['date_block_num','shop_id','item_id','total_price_per_month',"avg_item_cnt_month"]])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


test['avg_price_per_month'] = y_test


# In[ ]:


test.head()


# **Predicting the item_cnt_month**

# In[ ]:


X = sales_train_shop_item[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month","min_price_per_month","max_item_cnt_month","max_price_per_month","avg_item_cnt_month","avg_price_per_month"]]
y = sales_train_shop_item["item_cnt_month"]
poly = PolynomialFeatures(degree=3)
X_ = poly.fit_transform(X)
reg = LinearRegression().fit(X_, y)
pred = poly.fit_transform(test[["date_block_num", "shop_id", "item_id","total_price_per_month","min_item_cnt_month","min_price_per_month","max_item_cnt_month","max_price_per_month","avg_item_cnt_month","avg_price_per_month"]])
y_test = reg.predict(pred)
print(y_test)


# In[ ]:


submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])


# In[ ]:


submit.head()


# In[ ]:


submit.to_csv('submission.csv',index = False)


# In[ ]:





# In[ ]:


# sample_submission = pd.read_csv(path+'sample_submission.csv')
# sample_submission.head()


# In[ ]:


# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from xgboost import XGBRegressor


# In[ ]:


# sales_train_shop_item.head()


# In[ ]:


# X = sales_train_shop_item[["date_block_num", "shop_id", "item_id"]]
# y = sales_train_shop_item["item_cnt_month"]
# # poly = PolynomialFeatures(degree=3)
# # X_ = poly.fit_transform(X)
# xgb_model = XGBRegressor(max_depth=8, 
#                          n_estimators=500, 
#                          min_child_weight=1000,  
#                          colsample_bytree=0.7, 
#                          subsample=0.7, 
#                          eta=0.3, 
#                          seed=0)
# xgb_model.fit(X, y, eval_metric="rmse")


# In[ ]:


# reg = LinearRegression().fit(X_, y)


# In[ ]:


# test.head(5)


# In[ ]:


# y_test = xgb_model.predict(test[['date_block_num','shop_id','item_id']])
# # pred = poly.fit_transform(test[['date_block_num','shop_id','item_id']])


# In[ ]:


# y_test = reg.predict(pred)


# In[ ]:


# y_test


# In[ ]:


# submit = pd.DataFrame({'ID':np.arange(len(y_test)),'item_cnt_month':np.clip(y_test, a_min = 0, a_max = 20)},columns=['ID','item_cnt_month'])


# In[ ]:


# submit.describe()


# In[ ]:


# submit.to_csv('submission.csv',index = False)

