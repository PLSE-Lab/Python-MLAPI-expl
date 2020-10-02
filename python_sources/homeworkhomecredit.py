#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install scikit-misc')


# In[ ]:


get_ipython().system('pip install scikit-learn==0.21.3')


# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_summary import DataFrameSummary
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
print('Import Succeed')

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("/kaggle/input/home-credit-default-risk/"))
PATH = "/kaggle/input/home-credit-default-risk/"


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


# In[ ]:


df_raw = pd.read_csv(f'{PATH}application_train.csv', low_memory = False)
df_test = pd.read_csv(f'{PATH}application_test.csv', low_memory = False)
df_raw.shape
train_cats(df_raw)
apply_cats(df_test, df_raw)
df, y, nas = proc_df(df_raw, 'TARGET')
test,_ , nas = proc_df(df_test, na_dict=nas)
print(df_raw.shape)
print(df_test.shape)
print(df.shape)
print(test.shape)
print(y.shape)


# In[ ]:


m = RandomForestRegressor(n_jobs=4, n_estimators=100)
m.fit(df, y)
m.score(df, y)
prediction = m.predict(test)


# In[ ]:


m.score(df, y)


# In[ ]:



submit = pd.DataFrame()
submit['SK_ID_CURR']=test.SK_ID_CURR
submit['TARGET']=prediction
submit.to_csv('submit.csv',index=False)

