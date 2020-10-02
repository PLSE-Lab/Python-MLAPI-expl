#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the currentd directory are saved as output.


# # Loading data
# 
# Loading data from train and test file. Test file provides only input data and I'll predict the prices via using a model.

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print(train_data.head()) 
print(test_data.head())

print(train_data.columns)
print(test_data.columns)

def get_numeric_cols(df):
    return [col for col in df.columns if df[col].dtype != 'object']

numeric_cols = get_numeric_cols(train_data)
train_data[numeric_cols] = Imputer().fit_transform(train_data[numeric_cols])
numeric_cols_test = get_numeric_cols(test_data)
test_data[numeric_cols_test] = Imputer().fit_transform(test_data[numeric_cols_test])

train_y = train_data.SalePrice
# test is meant for predictions and doesn't contain any price data. I need to provide it.

cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]
print('Missing:\n',cols_with_missing)

cand_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
cand_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

numeric_cols = [col for col in cand_train_predictors.columns if train_data[col].dtype != 'object']

low_cardinality = [col for col in cand_train_predictors if
                   train_data[col].nunique() < 10 and train_data[col].dtype == 'object']

predictor_columns = numeric_cols + low_cardinality

train_X = train_data[predictor_columns]
test_X = test_data[predictor_columns]

train_X=pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)
train_X, test_X = train_X.align(test_X,join='left', axis=1)

print('-'*80)
print('cols in train which contain nan')
print([col for col in train_X.columns if train_X[col].isnull().any()])
print('-'*80)
still_missing = [col for col in test_X.columns if test_X[col].isnull().any()]
print('cols in test which contain nan before fillna')
print(still_missing)
if(len(still_missing) > 0):
    test_X = test_X.fillna(0)
print('-'*80)
print('cols in test which contain nan after fillna')
print([col for col in test_X.columns if test_X[col].isnull().any()])
print('-'*80)
print('cols in test but not in train')
not_in_train = [col for col in test_X.columns if col not in train_X.columns]
print(not_in_train)        
print('-'*80)
print('cols in train but not in test')
not_in_test = [col for col in train_X.columns if col not in test_X.columns]
print(not_in_test)        


# # Model
# 
# Now I will fit a model.

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.fit_transform(val_X)

steps = np.linspace(0.001, 0.1, 100)
maes = []
for learning_rate in steps:
    my_model = XGBRegressor(n_estimators=1000, learning_rate=learning_rate, random_state=1)
    my_model.fit(train_X, train_y,
                 early_stopping_rounds=5,
                 eval_set=[(val_X, val_y)],
                 verbose=False)

    predictions = my_model.predict(val_X)
    maes.append(mean_absolute_error(val_y, predictions))
    print('mae[%f]:[%f] ' % (learning_rate, maes[-1]))
    
index = np.argmin(maes)
learning_rate = steps[index]
print('optimal learning rate:', learning_rate)


# # Predicting and submitting
# 
# Now it's time to predict from test.

# In[ ]:


test_X = test_X.as_matrix()

my_model = XGBRegressor(n_estimators=1000, learning_rate=learning_rate, random_state=1)
my_model.fit(train_X, train_y,
             early_stopping_rounds=5,
             eval_set=[(val_X, val_y)],
             verbose=False)

predicted_prices = my_model.predict(test_X)
print(predicted_prices[:5])

# print(len(predicted_prices))
# print(len(test_data.Id))

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.Id = my_submission.Id.astype(int)
# print(my_submission.Id)
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

