#!/usr/bin/env python
# coding: utf-8

# ## About: 
# This kernel aims to provide simplistic approch to solve timer series forecasting using the dataset provided by the largest Russian software firms- 1C Company [available here](http://www.kaggle.com/c/competitive-data-science-predict-future-sales/data). The data comprises of daily sales data acros various items and various shops over a span of 34 months. The script is heavily inspired from Dines's kerneral [available here](https://www.kaggle.com/dlarionov/feature-engineering-xgboost/notebook). My notebook simplifies some of the heavy feature engineering concepts and visualization of sales data.  
# #### Import Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cats = pd.read_csv('../input/item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')
# Set index  to ID
test = pd.read_csv('../input/test.csv').set_index('ID')


# In[ ]:


cats.head()


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


#Consider removing outliers


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set(xlabel = "Month", ylabel = "Sales")
ax.plot(train.date_block_num, train.item_cnt_day)
plt.show()


# #### Include first time occuring shop_id and item_id in test set to train set by imputing zero.

# In[ ]:


#len(set(test.item_id))
#len(set(train.item_id))
#len(set(test.item_id).intersection(set(train.item_id)))
len(set(test.item_id)) -(len(set(test.item_id).intersection(set(train.item_id))))


# In[ ]:


train.columns


# In[ ]:


train[train.date_block_num ==0]


# In[ ]:


from itertools import product
matrix = []
cols = ['date_block_num', 'shop_id', 'item_id']
# date_block_num = 1st to 33rd;
# train set has data from Jan 2013 to Oct 2015
# date_block_num = 34th  
# test set has data for Nov 2015
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), 
                                        sales.item_id.unique())), 
                                dtype = 'int16'))


# In[ ]:


matrix


# In[ ]:


matrix = pd.DataFrame(np.vstack(matrix), columns = cols)


# In[ ]:


matrix.head()


# In[ ]:


matrix.shape


# In[ ]:


matrix[matrix.date_block_num == 1].shape


# In[ ]:


matrix[matrix.date_block_num == 3].shape


# In[ ]:


matrix.info()


# In[ ]:


matrix.sort_values(cols, inplace = True)


# In[ ]:


train['revenue'] = train['item_price']* train['item_cnt_day']


# In[ ]:


group = train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':['sum'],
                                                                    'item_price':['mean']})
group.columns = ['item_cnt_month', 'avg_item_price_shopwise']
group.reset_index(inplace = True)
group.head()


# In[ ]:


#merge matrix and group
matrix = pd.merge(matrix, group, on=cols, how = 'left')


# In[ ]:


matrix.shape


# In[ ]:


matrix['item_cnt_month'] = (matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16))


# In[ ]:


matrix.shape


# #### Test set

# In[ ]:


# append time to test set
test['date_block_num'] = 34


# In[ ]:


test.columns


# In[ ]:


test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
#test['item_price']=0


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


#test.merge(train, how='left')
cols = ['shop_id', 'item_id', 'date_block_num']
#pd.merge(test,train, on=cols, how='left').fillna(0)
#test.join(train, on=cols)
#pd.merge(test, train, on=cols, how='left')
#test2 = pd.merge(test, matrix, on=['shop_id','item_id'], how='left')
#df2.drop_duplicates(subset=['A'])
matrix_tmp = matrix.groupby(['shop_id', 'item_id']).mean()['avg_item_price_shopwise'].reset_index()
test = pd.merge(test, matrix_tmp, on=['shop_id','item_id'], how='left')


# In[ ]:


test.shape


# In[ ]:


test.head()


# In[ ]:


test.isna().sum()


# In[ ]:


# Fill NaN with zero for avg_item_price
test['avg_item_price_shopwise'].fillna(0, inplace = True)


# In[ ]:



#test = test.dropna()


# In[ ]:


test.shape


# In[ ]:


#Drop item_cnt_month and avg_item_price_shopwise
#test = test.drop(['avg_item_price_shopwise_y'], axis=1)


# In[ ]:


#test=test.rename(columns = {'avg_item_price_shopwise_y':'avg_item_price_shopwise'})


# In[ ]:


matrix = pd.concat([matrix, test], ignore_index = True, sort = False, 
                  keys= cols)
matrix.fillna(0, inplace = True)


# In[ ]:


matrix.tail()


# In[ ]:


#Drop item_cnt_month and avg_item_price_shopwise
#matrix = matrix.drop(['avg_item_price_shopwise_x'], axis=1)


# In[ ]:


#matrix = matrix.drop(['item_cnt_month'], axis=1)


# In[ ]:


#Add category data to matrix
#matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')


# #### Target lags

# In[ ]:





# In[ ]:


matrix.head()


# In[ ]:


matrix.shape


# In[ ]:


def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] +=i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        return df


# In[ ]:


matrix = lag_feature(matrix, [1,2,3,6, 12], 'item_cnt_month')


# In[ ]:


matrix.head()


# In[ ]:


matrix.shape


# In[ ]:


#print(matrix.item_cnt_month_lag_1)


# In[ ]:


def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)


# In[ ]:


df = matrix.copy()
X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)


# In[ ]:


Y_train = df[df.date_block_num < 33]['item_cnt_month']


# In[ ]:


X_validate = df[df.date_block_num ==33].drop(['item_cnt_month'], axis=1) 


# In[ ]:


Y_validate = df[df.date_block_num ==33]['item_cnt_month']


# In[ ]:


X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)


# In[ ]:


from xgboost import XGBRegressor
from xgboost import plot_importance


# In[ ]:


model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,    
    seed=42)


# In[ ]:


model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_validate, Y_validate)], 
    verbose=True, 
    early_stopping_rounds = 10)


# In[ ]:


Y_pred = model.predict(X_validate).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)


# In[ ]:


submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)

