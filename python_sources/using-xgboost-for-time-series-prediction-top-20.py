#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <h1>[mlwhiz] Using XGBoost for time series prediction tasks</h1>
# 
# This notebook is a basic implementation of the content of the great article [by mlwhiz.com](http://mlwhiz.com/blog/2017/12/26/How_to_win_a_data_science_competition/) on how he got into the top 10 of 'Final Project' competition. 
# 
# **What you will find here?**
# 1.  A working notebook that you can submit that follows the idea of mlwhiz blog
# 2. Copy paste of his data-manipulation
# 
# **What you will not find here**
# 1.  A replication of his result: namely I just used a basic xgboost to make it work, what I am interested in is the data manipulation part
# 
# <h3>Description of the Problem:</h3>
# 
# In this competition we were given a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company.
# 
# We were asked you to predict total sales for every product and store in the next month.
# 
# The evaluation metric was RMSE where True target values are clipped into [0,20] range. This target range will be a lot important in understanding the submissions that I will prepare.
# 
# The main thing that I noticed was that the data preparation aspect of this competition was by far the most important thing. I creted a variety of features. Here are the steps I took and the features I created."

# In[ ]:


DATA_FOLDER = '../input'
transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))


# this are the transactions we will groupby from, + category_id

# In[ ]:


transactions = pd.merge(transactions, items, on='item_id', how='left')
transactions = transactions.drop('item_name', axis=1)
transactions.head()


# In[ ]:


from itertools import product
index_cols = ['shop_id', 'item_id', 'date_block_num']


# <h2>**1. Created a dataframe of all Date_block_num, Store and Item combinations:**</h2>
# This is important because in the months we don't have a data for an item store combination, the machine learning algorithm needs to be specifically told that the sales is zero.

# In[ ]:


grid = []
for block_num in transactions['date_block_num'].unique():
    cur_shops = transactions.loc[transactions['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = transactions.loc[transactions['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)


# Up to now we just crated a combinations of all possible triples (shop_id, item_id, date_block_num)

# In[ ]:


grid.head()


# <h2>3. Created Mean Encodings:</h2>

# <h2>A)</h2>
# So here we are grouping the transactions and applying the aggreated functions where ** sum of item_cnt_day will be our target variable** .
# 
# Why? This two new columns uses target values so they will be used 1) as a target value 2) to create lagged values in the time serie

# In[ ]:


mean_transactions = transactions.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum','item_price':np.mean}).reset_index()


# After calculating target variable and price_mean we merge it back to our grid, and we add item_category_id

# In[ ]:


mean_transactions = pd.merge(grid,mean_transactions,on=['date_block_num', 'shop_id', 'item_id'],how='left').fillna(0)


# In[ ]:


mean_transactions = pd.merge(mean_transactions, items, on='item_id',how='left')


# In[ ]:


mean_transactions.head()


# <h2>B)</h2>
# Here we create **MEAN ENCODING**
# 
# Now we create additional encoding with aggregation functions on our data as follow: [('**item_price**',np.mean,'**avg**'),('**item_cnt_day**',np.sum,'**sum**'),('**item_cnt_day**',np.mean,'**avg**')]:
# 

# In[ ]:


for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:
        
        mean_df = transactions.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']
        mean_transactions = pd.merge(mean_transactions, mean_df, on=['date_block_num',type_id], how='left')


# <h2>KEY POINT to understand this passage</h2>
# What the difference between A and B? (try to answer your self first)
# 
# 
# **Answer:** is in how we are grouping the data
# 
# <p>A) -> transactions.groupby(['date_block_num', 'shop_id', 'item_id'])</p>
# <p>B) -> transactions.groupby([type_id,'date_block_num'])</p>
# 
# In A) we are grouping our features by BOTH shop_id and item_id , in B) we group individually by shop_id, by item_id and by item_category_id. This follow a suggestion in the course on coursera.

# In[ ]:


mean_transactions.head(10)


# These above lines add the following 9 features :
# 
# 'item_id_avg_item_price'
# 'item_id_sum_item_cnt_day'
# 'item_id_avg_item_cnt_day'
# 'shop_id_avg_item_price',
# 'shop_id_sum_item_cnt_day'
# 'shop_id_avg_item_cnt_day'
# 'item_category_id_avg_item_price'
# 'item_category_id_sum_item_cnt_day'
# 'item_category_id_avg_item_cnt_day'

# <h2>4. Create Lag Features:</h2>

# What we do now? We created our mean encoding, but we can not using at prediction time (because we don't know the target variable so we can not encode it). What we do we create a lag for all this mean encoding features and we will use this lagged value (mean encoding at t-1, t-2 ..) to predict our values

# In[ ]:


lag_variables  = list(mean_transactions.columns[7:])+['item_cnt_day']
lags = [1, 2, 3, 6]
from tqdm import tqdm_notebook
for lag in tqdm_notebook(lags):

    sales_new_df = mean_transactions.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    mean_transactions = pd.merge(mean_transactions, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')


# As you can see below, now we have the same mean_encoded features but also lagged, the number of columns now is around 50

# In[ ]:


mean_transactions.head()


# <h2>5. Fill NA with zeros:</h2>

# In[ ]:


mean_transactions = mean_transactions[mean_transactions['date_block_num']>12]


# In[ ]:


for feat in mean_transactions.columns:
    if 'item_cnt' in feat:
        mean_transactions[feat]=mean_transactions[feat].fillna(0)
    elif 'item_price' in feat:
        mean_transactions[feat]=mean_transactions[feat].fillna(mean_transactions[feat].median())


# <h2>IMPORTANT LINE TO UNDERSTAND HOW TO TRAIN THE MODEL</h2>
# in the next line we choose to 'drop' some columns, what are these columns? are the one that we will not be able to have at prediction time. 
# What column we won't have a prediction time? All the column that are **not lagged**, thats why we drop all lag_variables
# 
# [note: for me this was the key to understand everything]

# In[ ]:


cols_to_drop = lag_variables[:-1] + ['item_price', 'item_name'] # dropping all target variables but not "item_cnt_day" cause is target


# In[ ]:


training = mean_transactions.drop(cols_to_drop,axis=1)


# <h2>6. Now we can finally train the XGB model</h2>
# I wrote a really simple xgb process to train a model with the given data model, training the xgb will takes more or less 10 mins

# In[ ]:


import xgboost as xgb


# In[ ]:


xgbtrain = xgb.DMatrix(training.iloc[:, training.columns != 'item_cnt_day'].values, training.iloc[:, training.columns == 'item_cnt_day'].values)


# In[ ]:


param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'} # random parameters
bst = xgb.train(param, xgbtrain)


# In[ ]:


x=xgb.plot_importance(bst)
x.figure.set_size_inches(10, 30) 


# the most important features (2 0 5 8 4 1 3 9 33) are:

# In[ ]:


cols = list(training.columns)
del cols[cols.index('item_cnt_day')] # eliminate target feature col name


# In[ ]:


[cols[x] for x in [2, 0, 5, 8, 4, 1, 3, 9, 33]]


# <h2>7. Preparing predictions</h2>
# now we need to embrace the hard task of preparing prediction data without bugs

# In[ ]:


training.columns


# so we need these columns above without 'item_cnt_day', and we have only the column below

# In[ ]:


test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv.gz'))
test.head()


# So we need to manipulate the training set similarly to how we did in the first part of the notebook.
# - add date_block_num = 34
# - add category_id
# - add lagging

# In[ ]:


test['date_block_num'] = 34


# In[ ]:


test = pd.merge(test, items, on='item_id', how='left')


# In[ ]:


from tqdm import tqdm_notebook
for lag in tqdm_notebook(lags):

    sales_new_df = mean_transactions.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    test = pd.merge(test, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')


# **They look the same!**

# In[ ]:


_test = set(test.drop(['ID', 'item_name'], axis=1).columns)
_training = set(training.drop('item_cnt_day',axis=1).columns)
for i in _test:
    assert i in _training
for i in _training:
    assert i in _test


# In[ ]:


assert _training == _test


# In[ ]:


test = test.drop(['ID', 'item_name'], axis=1)


# In[ ]:


for feat in test.columns:
    if 'item_cnt' in feat:
        test[feat]=test[feat].fillna(0)
    elif 'item_price' in feat:
        test[feat]=test[feat].fillna(test[feat].median())


# lets check that our lag is actually correct

# In[ ]:


test[['shop_id','item_id']+['item_cnt_day_lag_'+str(x) for x in [1,2,3]]].head()


# In[ ]:


print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 33]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 32]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 31]['item_cnt_day'])


# the lagged value for (5	5037) actually correspond, looks like we dont have bugs!

# <h2>**8. Predict**

# In[ ]:


xgbpredict = xgb.DMatrix(test.values)


# In[ ]:


pred = bst.predict(xgbpredict)


# In[ ]:


pd.Series(pred).describe()


# In[ ]:


pred = pred.clip(0, 20)


# In[ ]:


pred.sum()


# In[ ]:


pd.Series(pred).describe()


# In[ ]:


sub_df = pd.DataFrame({'ID':test.index,'item_cnt_month': pred })


# In[ ]:


sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv',index=False)


# In[ ]:




