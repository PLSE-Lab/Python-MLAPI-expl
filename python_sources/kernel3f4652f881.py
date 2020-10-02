#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import urllib
from io import BytesIO
from torch.utils.data import Dataset
import os
from tempfile import TemporaryDirectory

from fastai.vision import * 
from fastai import *
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from PIL import Image



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_url = 'https://www.dropbox.com/s/bjtilcuhlmvs81f/LOGD_training_disguisedCLEANED50ANDCorrToTarget.csv?dl=1'  
test_url = 'https://www.dropbox.com/s/p99ltb659qt1suk/LOGD_test_disguised.csv?dl=1'


# In[ ]:


df_train = pd.read_csv(train_url)
df_test = pd.read_csv(test_url)


# In[129]:


df_train.head()


# In[130]:


df_train.describe()


# In[ ]:


df_train = df_train.drop(['MOLECULE'], axis=1)


# In[ ]:


df_test = df_test.drop(['MOLECULE'], axis=1)


# In[ ]:


columns_test = df_test.columns.values.tolist()


# In[ ]:


columns_test[:10]


# In[ ]:


len(set(columns_test).intersection(columns))


# In[ ]:


columns = df_train.columns.values.tolist()
len(columns)


# In[ ]:


df_train[columns[12:13]].sum().values[0]


# In[ ]:


all_sum = sorted([(df_train[[columns[i]]].astype(bool).sum(axis=0).values[0], i) for i in range(2, len(columns))])


# In[ ]:


all_sum[:20]


# In[ ]:


to_drop = [e[1] for e in all_sum if e[0] < 1100]
len(to_drop)


# In[ ]:


pd.DataFrame({'app': list([e[0] for e in all_sum])}).hist(figsize=(15,12),bins = 20, color="#007959AA")


# In[ ]:


df_simple = df_train.drop([columns[i] for i in to_drop], axis=1)


# In[ ]:


set_test_columns = set(columns_test)


# In[ ]:


all(e in set_test_columns for e in df_simple.columns.values)


# In[ ]:


len(set(df_simple.columns.values.tolist()) - set_test_columns)


# In[ ]:


set(df_simple.columns.values.tolist()) - set_test_columns


# In[ ]:


df_simple = df_simple.drop(['D_12'], axis=1)


# In[ ]:


df_test_simple = df_test[df_simple.columns.values.tolist()]


# In[ ]:


df_train.isnull().values.any()


# In[ ]:


df_train.__len__()


# In[ ]:


from fastai.tabular import * 


# In[ ]:


train_range = int(df_train.__len__() * 0.75)
train_range


# In[ ]:


valid_idx = range(train_range, len(df_train))


# In[ ]:


len(valid_idx), len(df_train)


# In[ ]:


# df_norm = (df_train - df_train.mean()) / (df_train.max() - df_train.min())


# In[ ]:


df_norm_simple = (df_simple - df_simple.mean()) / (df_simple.max() - df_simple.min())


# In[ ]:


# df_norm.Act = df_train.Act


# In[ ]:


df_norm_simple.Act = df_train.Act


# In[ ]:


# df_norm.describe()


# In[ ]:



data = TabularDataBunch.from_df('/kaggle/working', df_norm_simple, dep_var='Act', valid_idx=valid_idx)


# In[ ]:


model = tabular_learner(data, layers=[200, 100], metrics=mean_squared_error)


# In[ ]:


model.fit_one_cycle(10, 1e-2)


# In[ ]:


import xgboost


# In[ ]:


model_xgb =  xgboost.XGBRegressor()


# In[ ]:


model_xgb.fit(df_simple, df_train['Act'])


# In[ ]:


y_pred = model_xgb.predict(df_test_simple)


# In[139]:


from sklearn.metrics import r2_score


# In[140]:


r2_score(df_test['Act'], y_pred)


# In[ ]:


df_simple['Act'] = df_train['Act']
df_test_simple['Act'] = df_test['Act']


# In[199]:


concat_data = pd.concat([df_simple, df_test_simple])
concat_data.__len__()


# In[200]:


df_simple.__len__()


# In[203]:


valid_idx = range(df_simple.__len__(), concat_data.__len__())


# In[204]:


df_concat_data_norm = (concat_data - concat_data.mean()) / (concat_data.max() - concat_data.min())


# In[205]:


df_concat_data_norm['Act'] = pd.concat([df_train['Act'], df_test['Act']])


# In[206]:


data2 = TabularDataBunch.from_df('/kaggle/working', df_concat_data_norm, dep_var='Act', valid_idx=valid_idx)


# In[163]:


from fastai.metrics import r2_score


# In[207]:


model_with_r2 = tabular_learner(data2, layers=[200, 150, 50], metrics=r2_score)


# In[208]:


model_with_r2.fit_one_cycle(10, 1e-2)


# In[209]:


model_with_r2.save('model_0758')


# In[ ]:




