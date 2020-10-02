#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/home-data-for-ml-course/train.csv')


# In[ ]:


train.describe()


# In[ ]:


train


# In[ ]:


y = train.SalePrice


# In[ ]:


y.describe()


# In[ ]:


X = train.drop(['Id', 'SalePrice'], axis=1)


# In[ ]:


X.describe()


# In[ ]:


X


# In[ ]:


from sklearn.model_selection import train_test_split
val_X, train_X, val_y, train_y = train_test_split(X,y, random_state=0, train_size=0.8, test_size=0.2)


# In[ ]:


val_X


# In[ ]:


train_X


# In[ ]:


s = (train.dtypes == 'object')
object_cols = list(s[s].index)
print('categorical data')
print(object_cols)


# In[ ]:


categorical_cols = [cname for cname in train_X if train_X[cname].nunique() <=5 and train_X[cname].dtype == 'object']


# In[ ]:


numerical_cols = [cname for cname in train_X.columns if train_X[cname].dtype in ['int64', 'float64']]


# In[ ]:


my_cols = categorical_cols + numerical_cols
X_train = train_X[my_cols].copy()
X_val = val_X[my_cols].copy()


# In[ ]:


X_train


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy= 'constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0, n_estimators = 200)


# In[ ]:


from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, train_y)

preds = my_pipeline.predict(X_val)
print('Mean Absolute Error:', mean_absolute_error(preds, val_y))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, X_train, X_val, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0, n_estimators=200)
    my_pipeline.fit(X_train, train_y)
    preds = my_pipeline.predict(X_val)
    mae = mean_absolute_error(preds, val_y)
    return (mae)


# In[ ]:


for max_leaf_nodes in [5, 35, 50, 65, 75, 100, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_val, train_y, val_y)
    print('mae: ',  max_leaf_nodes, my_mae)


# In[ ]:


# Test Data


# In[ ]:


test = pd.read_csv('../input/home-data-for-ml-course/test.csv')


# In[ ]:


test.describe()


# In[ ]:


test.head()


# In[ ]:


test


# In[ ]:


X_test = test.drop(['Id'], axis=1)


# In[ ]:


X_test


# In[ ]:


X_test.equals(X)


# In[ ]:


predicted_prizes = my_pipeline.predict(X_test)


# In[ ]:


predicted_prizes


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prizes})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




