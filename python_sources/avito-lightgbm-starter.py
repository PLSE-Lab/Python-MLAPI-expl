#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import gc
import lightgbm
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


cols = ['parent_category_name', 'category_name', 'price', 'user_type', 'item_seq_number', 'image_top_1']
dummy_cols = ['parent_category_name', 'category_name','user_type']
y = train['deal_probability'].copy()
x_train = train[cols].copy().fillna(0)
x_test  = test[cols].copy().fillna(0)
del train, test; gc.collect()

n = len(x_train)
x = pd.concat([x_train, x_test])
x = pd.get_dummies(x, columns=dummy_cols)
x.head()


# In[ ]:


x_train = x.iloc[:n, :]
x_test = x.iloc[n:, :]
del x; gc.collect()


# In[ ]:


# https://www.kaggle.com/ezietsman/simple-python-lightgbm-example

# Create training and validation sets
x, x_val, y, y_val = train_test_split(x_train, y, test_size=0.2, random_state=42)

# Create the LightGBM data containers
train_data = lightgbm.Dataset(x, label=y)
val_data = lightgbm.Dataset(x_val, label=y_val)

# Train the model
parameters = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 100
}


model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=val_data,
                       num_boost_round=2000,
                       early_stopping_rounds=100)

# Create a submission
y_pred = model.predict(x_test)


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)


# In[ ]:


sub.to_csv('simple_mean_benchmark.csv', index=False)
sub.head()

