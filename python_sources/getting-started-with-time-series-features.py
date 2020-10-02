#!/usr/bin/env python
# coding: utf-8

# ### A very basic kernel with some naive features for predicting seasonal or hierachical autoregressive problems (such as store sales)
# * Kernel will be updated
# 

# In[ ]:


import pandas as pd
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')


# ### merge train and test for easier feature engineering. 
# * Beware leaks and the offset needed in predicting the future!!
# *  Can also be useful for creating a baseline; see: https://machinelearningmastery.com/model-residual-errors-correct-time-series-forecasts-python/
# *  Note that here, train has no ID column
# 

# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


print("train shape:", train.shape)
print("Test shape:", test.shape)
df = pd.concat([train,test])
print(df.shape)
df.head()


# ### naive datetime features:
# * Could also add holidays, weekends, work-hours if relevant and known

# In[ ]:


df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)


df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.dayofweek
df['year'] = df['date'].dt.year
# df['date'].dt.
df['week_of_year']  = df.date.dt.weekofyear


# In[ ]:


df.set_index("date",inplace=True)


# ## Add historical / seasonal features
# * Additional features could includes slopes, trends, item-basket level features, but require more work to avoid leakage. Will add in future
# * For now -  naive features: we expect sales to be what they were at the same time of year, in the past, for each store+item combo
#     * Could be done more elegantly with pandas's agg_func

# In[ ]:


df["median-store_item-month"] = df.groupby(['month',"item","store"])["sales"].transform("median")
df["mean-store_item-week"] = df.groupby(['week_of_year',"item","store"])["sales"].transform("mean")
df["item-month-sum"] = df.groupby(['month',"item"])["sales"].transform("sum") # total sales of that item  for all stores
df["store-month-sum"] = df.groupby(['month',"store"])["sales"].transform("sum") # total sales of that store  for all items


# In[ ]:


# get shifted features for grouped data. Note need to sort first! 
df['store_item_shifted-90'] = df.groupby(["item","store"])['sales'].transform(lambda x:x.shift(90)) # sales for that item 90 days = 3 months ago
df['store_item_shifted-180'] = df.groupby(["item","store"])['sales'].transform(lambda x:x.shift(180)) # sales for that item 180 days = 3 months ago
df['store_item_shifted-365'] = df.groupby(["item","store"])['sales'].transform(lambda x:x.shift(365)) # sales for that 1 year  ago

df["item-week_shifted-90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).sum()) # shifted total sales for that item 12 weeks (3 months) ago
df["store-week_shifted-90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).sum()) # shifted total sales for that store 12 weeks (3 months) ago
df["item-week_shifted-90"] = df.groupby(['week_of_year',"item"])["sales"].transform(lambda x:x.shift(12).mean()) # shifted mean sales for that item 12 weeks (3 months) ago
df["store-week_shifted-90"] = df.groupby(['week_of_year',"store"])["sales"].transform(lambda x:x.shift(12).mean()) # shifted mean sales for that store 12 weeks (3 months) ago


# In[ ]:


df.tail()


# ## We should do one hot encoding at this point on the store and ite mIDs to avoid silly range based features. 
# * We'll do that later, as it's also possible the numbers/order has meaning. 

# In[ ]:





# ## split our data for modelling

# In[ ]:


col = [i for i in df.columns if i not in ['date','id']]
y = 'sales'


# In[ ]:


train.columns


# In[ ]:


train = df.loc[~df.sales.isna()]
print("new train",train.shape)
test = df.loc[df.sales.isna()]
print("new test",test.shape)


# # Evaluation should use **temporal train test** split or temporal CV
# 
# *  we can define it manually or use sklearn's functions. these aren't trivial to plug and play with xgboost, so i'll skip for this version of the kernel, but without it, our local score is meaningless!

# In[ ]:


train_x, train_cv, y, y_cv = train_test_split(train[col],train[y], test_size=0.15, random_state=42)
# train_x, train_cv, y, y_cv = TimeSeriesSplit(train[col],train[y], test_size=0.1, random_state=42)


# In[ ]:


def XGB_regressor(train_X, train_y, test_X, test_y, feature_names=None, seed_val=2017, num_rounds=500):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model    


# In[ ]:


model = XGB_regressor(train_X = train_x, train_y = y, test_X = train_cv, test_y = y_cv)
y_test = model.predict(xgb.DMatrix(test[col]), ntree_limit = model.best_ntree_limit)


# In[ ]:


sample['sales'] = y_test
sample.to_csv('simple_starter.csv', index=False)

