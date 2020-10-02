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
import xgboost as xgb
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


# ### Libraries

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# ###### Joining test and train data
# 

# In[ ]:


df=pd.concat([train,test])
df.head()


# In[ ]:


df.isnull().sum()


# ###### Changing date datatype

# In[ ]:


df['date']=pd.to_datetime(df['date'])


# In[ ]:


df.dtypes


# > ***New columns***

# In[ ]:


def cols_new(data_df):
    data_df['year'] = data_df['date'].dt.year
    data_df['quarter'] = data_df['date'].dt.quarter
    data_df['month'] = data_df['date'].dt.month
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    #data_df['weekday'] = data_df['date'].dt.weekday
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    return data_df


# In[ ]:


cols_new(df)


# In[ ]:


df.groupby(['item','store'])['sales'].median()


# In[ ]:


df.columns


# ## Mean Function 
# groupby two columns and mean of sales*

# In[ ]:


get_ipython().run_cell_magic('time', '', "def mean_cols(data,cols):\n   for i in cols:\n       cols=[e for e in cols if e not in (i)]\n       for j in cols :\n           if i!=j :\n               data['mean_'+i+'_'+j]=data.groupby([i,j])['sales'].transform('mean')\n   return data")


# In[ ]:


df.columns


# In[ ]:


get_ipython().run_cell_magic('time', '', "mean_cols(df,['item','store','dayofweek','weekofyear','month','quarter'])\nprint(df.columns)")


# In[ ]:


df.shape


# ## Median Function 
# groupby two columns and median of sales

# In[ ]:


def median_cols(data,cols):
    for i in cols:
        cols=[e for e in cols if e not in (i)]
        for j in cols :
            if i!=j :
                data['median_'+i+'_'+j]=data.groupby([i,j])['sales'].transform('median')
    return data


# In[ ]:


get_ipython().run_cell_magic('time', '', "median_cols(df,['item','store','dayofweek','weekofyear','month','quarter'])\nprint(df.columns)")


# In[ ]:


df.shape


# In[ ]:


df.head()


# ###### saperating train and test

# In[ ]:


train = df.loc[~df.sales.isna()]
test = df.loc[df.sales.isna()]


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


train.isnull().sum().sum()


# ###### droping cols

# In[ ]:


X_train = train.drop(['date','sales','id'], axis=1)
y_train = train['sales'].values
X_test = test.drop(['id','date','sales'], axis=1)


# In[ ]:


X_train.isnull().sum().sum()


# ### Train Test split

# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(X_train, y_train, random_state=100, test_size=0.25)


# ## XGBOOST Algorithm 
# parameters declaring and model building

# In[ ]:


get_ipython().run_cell_magic('time', '', "params = {\n    'colsample_bytree': 0.8,\n    'eta': 0.1,\n    'eval_metric': 'mae',\n    'lambda': 1,\n    'max_depth': 6,\n    'objective': 'reg:linear',\n    'seed': 0,\n    'silent': 1,\n    'subsample': 0.8,\n}\nxgbtrain = xgb.DMatrix(x_train, label=y_train)\nxgbvalidate = xgb.DMatrix(x_validate, label=y_validate)\nxgbmodel = xgb.train(list(params.items()), xgbtrain, early_stopping_rounds=50,\n                     evals=[(xgbtrain, 'train'), (xgbvalidate, 'validate')], \n                     num_boost_round=200, verbose_eval=50)")


# ### Predicting

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = xgbmodel\n\n\npredict=pd.DataFrame(model.predict(xgb.DMatrix(X_test),ntree_limit=model.best_ntree_limit),columns=['sales'])")


# ### Submitting result

# In[ ]:


ids=pd.read_csv("../input/test.csv",usecols=['id'])
predict=np.round(predict)
sub=ids.join(predict)
sub.head()


# In[ ]:


sub.to_csv('xgb_grpby_mean_median.csv',index=False)


# In[ ]:




