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


# In[ ]:


import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# In[ ]:


train_csv = pd.read_csv("../input/train_set.csv", low_memory=False)
test_csv = pd.read_csv("../input/test_set.csv", low_memory=False)


# In[ ]:


X_trainvalid = train_csv.drop(["PRICE", "Id"], axis=1)
X_test = test_csv.drop(["Id"], axis=1)


# In[ ]:


X_traintest = pd.concat([X_trainvalid, X_test], axis=0)
X_traintest["SALEDATE"] = pd.to_datetime(X_traintest["SALEDATE"]).astype("int")
X_traintest["GIS_LAST_MOD_DTTM"] = pd.to_datetime(X_traintest["GIS_LAST_MOD_DTTM"]).astype("int")


# In[ ]:


categorical_features = [c for c in X_traintest.columns if c not in X_traintest.describe().columns] + ["USECODE"]


# In[ ]:


for c in categorical_features:
    X_traintest[c] = X_traintest[c].astype('category')


# In[ ]:


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 511,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'verbose': 0
}


# In[ ]:


X_train = X_traintest.iloc[:train_csv.shape[0], :]
X_test = X_traintest.iloc[train_csv.shape[0]:, :]

y_train = np.log(train_csv["PRICE"].values)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=71)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_features)


print('Starting training...')
# train
model = lgb.train(
    params, 
    lgb_train, 
    valid_sets=lgb_valid, 
    num_boost_round=10000, 
    early_stopping_rounds=100
)


# In[ ]:


test_pred = np.exp(model.predict(X_test))


# In[ ]:


sub_df = pd.DataFrame({"Id":test_csv["Id"].values,"PRICE":test_pred})
sub_df.to_csv("sub_harada.csv", index=False)


# In[ ]:




