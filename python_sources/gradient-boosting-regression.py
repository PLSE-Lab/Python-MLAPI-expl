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


# Import training data
train = pd.read_csv("../input/train.csv")

# Examine data
print(train.shape)
print(train.head())

# Import test data
test = pd.read_csv("../input/test.csv")

print(test.shape)
print(test.head())


# I want to take a look at the data to find outliers and such.
# Code borrowed from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show


# Two pretty obvious outliers can be deleted.

# In[ ]:


train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)


# Before fitting a model to the data, missing values must be dealt with and the data needs to be divided properly.

# In[ ]:


from sklearn import preprocessing

# Single out the features
train_features = train.loc[:, 'MSSubClass':'SaleCondition']
test_features = test.loc[:, 'MSSubClass':'SaleCondition']

# Group data to ensure same dimensions post one-hot encoding
all_data = pd.concat([train_features, test_features])

all_data = pd.get_dummies(all_data)

# Handle missing values
imputer = preprocessing.Imputer()

all_data = imputer.fit_transform(all_data)

# Separate data
train_features = all_data[:train.shape[0]]
test_features = all_data[train.shape[0]:]

train_target = train[['SalePrice']]

print(train_features.shape)
print(test_features.shape)


# Now a gradient boosting regression model can be fit to the data. Hyperparameters were determined experimentally.

# In[ ]:


from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from math import sqrt

model = ensemble.GradientBoostingRegressor(n_estimators=15000, max_depth=4, min_samples_leaf=15, min_samples_split=10, learning_rate=0.01, loss='huber', random_state=5)

# Reshape train_target to be a 1d array
train_target = train_target.as_matrix().flatten()

# Fit model
model.fit(train_features, train_target)


# Finally I'll make predictions on the test data provided.

# In[ ]:


# Make predictions with model
target_predictions = model.predict(test_features)

target_predictions = np.reshape(target_predictions, -1)

# Prepare solution
solution = pd.DataFrame({"id":test.Id, "SalePrice":target_predictions})

solution.to_csv('submission.csv', index=False)

