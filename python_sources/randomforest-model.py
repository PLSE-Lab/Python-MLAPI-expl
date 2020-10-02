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


train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.isna().sum()


# In[ ]:


train_data.dropna(axis=1, inplace=True)
test_data.dropna(axis=1, inplace = True)
train_data_cols = train_data.columns
test_data_cols = test_data.columns


# In[ ]:


print(train_data_cols)
print(test_data_cols)
colsToDrop = list(set(train_data_cols) - set(test_data_cols))
print(colsToDrop)


# In[ ]:


X = train_data.drop(colsToDrop, axis=1)
y = train_data['SalePrice']


# In[ ]:


test_data.drop(colsToDrop, axis=1, inplace=True, errors = 'ignore')


# In[ ]:


test_data.drop(['Electrical'], axis=1, inplace=True)


# In[ ]:


cols = X.columns
print(cols)
num_cols = X._get_numeric_data().columns
print(num_cols)
cat_cols = list(set(cols)-set(num_cols))
print(cat_cols)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for i in cat_cols:
    X[i] = le.fit_transform(X[i])

le_test = LabelEncoder()
for i in cat_cols:
    test_data[i] = le.fit_transform(test_data[i])


# In[ ]:


X.drop(['Id'], axis=1, inplace=True)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf_HomeModel = RandomForestRegressor(criterion='mae', n_estimators= 100)
rf_HomeModel.fit(X, y)


# In[ ]:


rf_HomeModel_pred = rf_HomeModel.predict(test_data.iloc[:,1:])


# In[ ]:


output = pd.DataFrame({'Id': test_data['Id'],
                      'SalePrice': rf_HomeModel_pred})


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:




