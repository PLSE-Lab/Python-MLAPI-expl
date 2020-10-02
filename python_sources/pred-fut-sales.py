#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# loading all necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import os

from itertools import product
from tqdm import tqdm_notebook

print(os.listdir("../input"))


# In[ ]:


# loading competition data
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
sales = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# sales are groupedby category_id
sales = pd.merge(sales, items, on='item_id', how='left')
sales = sales.drop('item_name', axis=1)
sales.head()


# In[ ]:


index_cols = ['shop_id', 'item_id', 'date_block_num']


# In[ ]:


# creating df of all item and store combos in order to assure the ML algorithim knows the sales are 0
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)


# In[ ]:


# display the grid and make observations that determine if im ready to move onto the next step
grid.head()


# In[ ]:


# preparing to create mean encodings by grouping the sales in order to create much needed lagged values for this time-series problem
mean_sales = sales.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum','item_price':np.mean}).reset_index()


# In[ ]:


#filling in NAs with 0s
mean_sales = pd.merge(grid,mean_sales,on=['date_block_num', 'shop_id', 'item_id'],how='left').fillna(0)


# In[ ]:


# merge back to grid
mean_sales = pd.merge(mean_sales, items, on='item_id',how='left')


# In[ ]:


mean_sales.head()


# In[ ]:


# actually creating the mean encoding feature now
# added a total of 9 features to further develop and prepare the data
for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:
# grouping indivudally by shop, item, and category id which was hinted at during the lectures from the course        
        mean_df = sales.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']
        mean_sales = pd.merge(mean_sales, mean_df, on=['date_block_num',type_id], how='left')


# In[ ]:


mean_sales.head(25)


# In[ ]:


# creating lag features in order to achieve a lag value which is crucial to predictting
lag_variables  = list(mean_sales.columns[7:])+['item_cnt_day']
lags = [1, 2, 3, 6]
for lag in tqdm_notebook(lags):
# this should keep the same mean encoded features that are needed but lagging them
    sales_new_df = mean_sales.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    mean_sales = pd.merge(mean_sales, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')


# In[ ]:


mean_sales.head()


# In[ ]:


# filling all of the N/As with 0s again
mean_sales = mean_sales[mean_sales['date_block_num']>12]


# In[ ]:


# filling N/As with 0s cont.
for feat in mean_sales.columns:
    if 'item_cnt' in feat:
        mean_sales[feat]=mean_sales[feat].fillna(0)
    elif 'item_price' in feat:
        mean_sales[feat]=mean_sales[feat].fillna(mean_sales[feat].median())


# In[ ]:


# dropping all of the non-lagged columns
cols_to_drop = lag_variables[:-1] + ['item_price', 'item_name']


# In[ ]:


training = mean_sales.drop(cols_to_drop,axis=1)


# In[ ]:


# begin the training of the model!
# xgboost seemed to me to be the most time efficient while remaining the most accurate and percise so i settled on it
# as well as it making the most sense to me through lectures
xgbtrain = xgb.DMatrix(training.iloc[:, training.columns != 'item_cnt_day'].values, training.iloc[:, training.columns == 'item_cnt_day'].values)


# In[ ]:


# specifying parameters to train and eventually obtain a predcition from...these were selected specifically so they would not crash the jupyter notebook
param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}
bst = xgb.train(param, xgbtrain)


# In[ ]:


x=xgb.plot_importance(bst)
x.figure.set_size_inches(10, 20) 


# In[ ]:


cols = list(training.columns)
del cols[cols.index('item_cnt_day')] # eliminate target feature col name


# In[ ]:


# based on xgb.plot_importance it is clear these are the most important features and were selected for that very reason
[cols[x] for x in [5, 8, 0, 2, 3, 4, 9, 33, 23]]


# In[ ]:


# prep the predictions data
training.columns


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


# add the month which we are trying to predict sales for
test['date_block_num'] = 34


# In[ ]:


# adding category ids
test = pd.merge(test, items, on='item_id', how='left')


# In[ ]:


# lastly, adding lagging to our prediction
from tqdm import tqdm_notebook
for lag in tqdm_notebook(lags):

    sales_new_df = mean_sales.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    test = pd.merge(test, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')


# In[ ]:


# now time to validate the lag
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


# In[ ]:


test[['shop_id','item_id']+['item_cnt_day_lag_'+str(x) for x in [1,2,3]]].head()


# In[ ]:


print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 33]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 32]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 31]['item_cnt_day'])


# squashed all of the bugs!

# In[ ]:


# Predicition Time!
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

