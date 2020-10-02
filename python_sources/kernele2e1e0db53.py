#!/usr/bin/env python
# coding: utf-8

# In[104]:


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

#Data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix

from xgboost import XGBClassifier


# In[79]:


data_full = pd.read_csv("../input/weatherAUS.csv")


# In[80]:


print(data_full.columns) # Get the column names
print(data_full.dtypes)


# In[81]:


data_full.RainTomorrow.isna().value_counts() # Make sure all our target data is present


# In[82]:


X_full = data_full.drop(["Date", "RISK_MM", "RainTomorrow"], axis = 1) # drop target and data leak (RISK_MM)
y_full = data_full.RainTomorrow # target


# In[83]:


X = pd.get_dummies(X_full)


# In[84]:


imputer = Imputer(strategy="most_frequent")
X = pd.DataFrame(imputer.fit_transform(X), columns = X.columns)


# In[85]:


X_train, x_valid, y_train, y_valid = train_test_split(X, y_full, random_state = 1, test_size=0.1)


# In[88]:


clf = XGBClassifier(n_estimators=500)
clf.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(x_valid, y_valid)])


# In[89]:


y_pred = clf.predict(x_valid)


# In[107]:


accuracy_score(y_valid, y_pred)


# 

# In[108]:


confusion_matrix(y_valid, y_pred)

