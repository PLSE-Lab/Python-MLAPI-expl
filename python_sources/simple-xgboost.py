#!/usr/bin/env python
# coding: utf-8

# ### Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
#import xgboost as xgb
#from sklearn.model_selection import train_test_split


# > **loading dataset**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.dtypes


# In[ ]:


train_df.head()


# In[ ]:


test_df.tail()


# In[ ]:


train_df['trsin_or_test'], test_df['trsin_or_test'] = 'train', 'test'
data_df = pd.concat([train_df, test_df])
data_df.head()


# In[ ]:


data_df['date']=pd.to_datetime(data_df['date'])


# In[ ]:


data_df.dtypes


# In[ ]:


data_df.info()


# In[ ]:


data_df['year'] = data_df['date'].dt.year
data_df['quarter'] = data_df['date'].dt.quarter
data_df['month'] = data_df['date'].dt.month
data_df['weekofyear'] = data_df['date'].dt.weekofyear
data_df['weekday'] = data_df['date'].dt.weekday
data_df['dayofweek'] = data_df['date'].dt.dayofweek


# In[ ]:


data_df.head()


# In[ ]:


data_df.groupby(['quarter', 'item'])['sales'].mean()


# > **New cols for mean() based on quarter**

# In[ ]:


data_df['item_quarter_mean'] = data_df.groupby(['quarter', 'item'])['sales'].transform('mean')


# In[ ]:


data_df.head()


# In[ ]:


data_df['store_quarter_mean'] = data_df.groupby(['quarter', 'store'])['sales'].transform('mean')
data_df['store_item_quarter_mean'] = data_df.groupby(['quarter', 'item', 'store'])['sales'].transform('mean')


# > **New cols for mean() based on month** 

# In[ ]:


data_df['item_month_mean'] = data_df.groupby(['month', 'item'])['sales'].transform('mean')
data_df['store_month_mean'] = data_df.groupby(['month', 'store'])['sales'].transform('mean')
data_df['store_item_month_mean'] = data_df.groupby(['month', 'item', 'store'])['sales'].transform('mean')


# > **New cols for mean() based on weekof year** 

# In[ ]:


data_df['item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item'])['sales'].transform('mean')
data_df['store_weekofyear_mean'] = data_df.groupby(['weekofyear', 'store'])['sales'].transform('mean')
data_df['store_item_weekofyear_mean'] = data_df.groupby(['weekofyear', 'item', 'store'])['sales'].transform('mean')


# > **New cols for mean() based on weekday** 

# In[ ]:


data_df['itemweekday_mean'] = data_df.groupby(['weekday', 'item'])['sales'].transform('mean')
data_df['storeweekday_mean'] = data_df.groupby(['weekday', 'store'])['sales'].transform('mean')
data_df['storeitemweekday_mean'] = data_df.groupby(['weekday', 'item', 'store'])['sales'].transform('mean')


# In[ ]:


data_df.head()


# In[ ]:


data_df.tail()


# In[ ]:


data_df.isnull().sum().sum()


# In[ ]:


data_df.info()


# In[ ]:


data_df.head()


# > **Model Predection**

# In[ ]:


data_df.shape


# In[ ]:


data_df.columns


# In[ ]:


data_df.drop(['date','id','sales'],axis=1,inplace=True)


# In[ ]:


data_df.info()


# In[ ]:


x= data_df[data_df['trsin_or_test'] == 'train']#.dropna().drop(['id', 'sales', 'trsin_or_test', 'date'], axis=1)
test = data_df[data_df['trsin_or_test'] == 'train']#.dropna()['sales']


# In[ ]:


x.head()


# In[ ]:


test.head()


# In[ ]:


x.drop(['trsin_or_test'],axis=1,inplace=True)
test.drop(['trsin_or_test'],axis=1,inplace=True)


# In[ ]:


y=pd.read_csv('../input/train.csv',usecols=['sales'])
y=y.sales


# In[ ]:


y.shape


# In[ ]:


y.head()


# x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=0, test_size=0.25)
# 
# print(x_train.shape, x_validate.shape, y_train.shape, y_validate.shape)
# 
#  %%time
# params = {
#     'colsample_bytree': 0.8,
#     'eta': 0.1,
#     'eval_metric': 'mae',
#     'lambda': 1,
#     'max_depth': 6,
#     'objective': 'reg:linear',
#     'seed': 0,
#     'silent': 1,
#     'subsample': 0.8,
# }
# xgbtrain = xgb.DMatrix(x_train, label=y_train)
# xgbvalidate = xgb.DMatrix(x_validate, label=y_validate)
# xgbmodel = xgb.train(list(params.items()), xgbtrain, early_stopping_rounds=50, evals=[(xgbtrain, 'train'), (xgbvalidate, 'validate')], num_boost_round=200, verbose_eval=50)

# In[ ]:


from sklearn import ensemble
xbr=ensemble.GradientBoostingRegressor()
xbr


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xbr.fit(x,y)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xbr.score(x,y)')


# > **submssion of results**
# 
# 

# %%time
# model = xgbmodel
# 
# 
# predict=pd.DataFrame(model.predict(xgb.DMatrix(test),ntree_limit=model.best_ntree_limit))
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "predict=pd.DataFrame(xbr.predict(test),columns=['sales'])")


# In[ ]:


ids=pd.read_csv("../input/test.csv",usecols=['id'])
sub=ids.join(predict)
sub.head()


# In[ ]:


sub.to_csv('sample.csv',index=False)

