#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv",parse_dates=["datetime"])
train.head()


# In[ ]:


test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv",parse_dates=["datetime"])
test.head()


# In[ ]:


train.info()
test.info()


# In[ ]:


train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
train.shape


# In[ ]:


test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
test.shape


# In[ ]:


train.head()


# In[ ]:


categorical_feature = ["season","holiday","workingday","weather","dayofweek","month","year","hour"]


# In[ ]:


for var in categorical_feature:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")
train.info()


# In[ ]:


feature = ["season","holiday","workingday","weather","dayofweek","year","hour","temp","atemp","humidity"]


# In[ ]:


X_train = train[feature]
X_test = test[feature]
X_train.head()


# In[ ]:


Y_train = train["count"]
Y_train.head()


# In[ ]:


model = RandomForestRegressor(n_estimators=500)

Y_train_log = np.log1p(Y_train)
model.fit(X_train,Y_train_log)

result = model.predict(X_test)


# In[ ]:


np.exp(result)


# In[ ]:


sub = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")
sub.head()


# In[ ]:


sub["count"] = np.exp(result)
sub.head()


# In[ ]:


sub.to_csv("20_03_29sub.csv",index=False)


# In[ ]:


160/3251

