#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Reading the files
train = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')
data_dict = pd.read_excel('../input/zillow_data_dictonary.xlsx')


# In[ ]:


print("Train Columns")
print(train.columns)
print("Properties Columns")
print(prop.columns)
print("Zillow Data Dictionary Columns")
print(data_dict.columns)


# In[ ]:


# Sample data's
print("The length of train", len(train))
train.head()


# In[ ]:


print("The length of train", len(prop))
prop.head()


# In[ ]:


print("The length of train", len(data_dict))
data_dict.head()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[ ]:


# Transcations distributions
(train['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# In[ ]:


(prop['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# In[ ]:


# Now before exploring further we will create a train set
train = pd.merge(train, prop, on='parcelid', how='left')


# In[ ]:


# The entire columns and also types
dtype = train.dtypes.reset_index()
dtype.columns = ["Name", "Column Type"]
dtype


# In[ ]:


# Now we will be building a model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[ ]:


# Get all the features ready
final  = [train['logerror']]
train.drop('logerror', axis=1, inplace=True)
for col in train.columns:
    print(col)
    if train[col].dtype == 'object':
        final.append(pd.get_dummies(train[col], prefix=col,drop_first=True))
    else:
        final.append(train[col])
target = pd.concat(final, axis=1)


# In[ ]:


train, test = train_test_split(target)
x_label = train['logerror']
x_train = train.drop('logerror', axis=1)

y_label = test['logerror']
y_train = test.drop('logerror', axis=1)
# fit model no training data
model = XGBRegressor()
model.fit(x_train, x_label)


# In[ ]:


# Now we will see variable importances
print(model.feature_importances_)
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
from xgboost import plot_importance
plot_importance(model, max_num_features=20)


# In[ ]:


# Tuning the paraemters for Xgboost we will use gridsearch along with 
from sklearn.model_selection import GridSearchCV
import copy
param_test2b = {
 'min_child_weight':[6,8,10,12],
 'learning_rate':[1, .1, .001],
  'n_estimators': [100, 200, 300],
   'max_depth': [3, 5, 6]
}

gsearch2b = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=140, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2b, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)
gsearch2b.fit(x_train,x_label.values)

