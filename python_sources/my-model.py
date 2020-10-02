#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **This will be your workspace for Kaggle's Machine Learning education track.**
# 
# You will build and continually improve a model to predict housing prices as you work through each tutorial.  Fork this notebook and write your code in it.
# 
# The data from the tutorial, the Melbourne data, is not available in this workspace.  You will need to translate the concepts to work with the data in this notebook, the Iowa data.
# 
# Come to the [Learn Discussion](https://www.kaggle.com/learn-forum) forum for any questions or comments. 
# 
# # Write Your Code Below
# 
# 

# In[ ]:


import pandas as pd

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# print(train_data.shape, test_data.shape)


# In[ ]:


# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor


# In[ ]:


train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
X = train_data.drop(['Id', 'SalePrice'], axis=1)
test = test_data.drop(['Id'], axis=1)


# In[ ]:


# one-hot encoding
categories_cols = [col for col in X.columns if X[col].nunique() < 10 and X[col].dtype == 'object']
numeric_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
my_cols = categories_cols + numeric_cols

X = X[my_cols]
test = test[my_cols]

one_hot_X = pd.get_dummies(X)
one_hot_test = pd.get_dummies(test)

final_onehot_X, final_onehot_test = one_hot_X.align(one_hot_test, join='left',
                                                       axis=1)


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(final_onehot_X.as_matrix(), y.as_matrix(),
                                                   test_size=0.25)

# impute missing values
my_imputer = Imputer()

final_onehot_imputed_X= my_imputer.fit_transform(final_onehot_X)
final_onehot_imputed_test = my_imputer.transform(final_onehot_test)

train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)


# In[ ]:


# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(train_X, train_y, early_stopping_rounds=5,
#             eval_set=[(test_X, test_y)], verbose=True)


# In[ ]:


my_model = XGBRegressor(n_estimators=143, learning_rate=0.05)
my_model.fit(final_onehot_imputed_X, y, verbose=False)

preds = my_model.predict(final_onehot_imputed_test)
print(preds)


# In[ ]:


# # drop houses where target is missing
# train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
# target = train_data.SalePrice
# train_X = train_data.drop(['Id','SalePrice'], axis=1)
# test_X = test_data.drop(['Id'], axis=1)

# # one-hot encoding
# categories_cols = [col for col in train_X.columns if train_X[col].nunique() < 10 and
#                   train_X[col].dtype == 'object']

# numeric_cols = [col for col in train_X.columns if train_X[col].dtype in ['int64', 'float64']]
# my_cols = categories_cols + numeric_cols

# train_X = train_X[my_cols]
# test_X = test_X[my_cols]

# one_hot_train_X = pd.get_dummies(train_X)

# cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

# # drop missing value
# train_X_drop_miss = train_X.drop(cols_with_missing, axis=1)

# # imputer missing values
# from sklearn.preprocessing import Imputer
# my_imputer = Imputer()
# one_hot_imputed_train_X = my_imputer.fit_transform(one_hot_train_X)


# In[ ]:


# one_hot_test_X = pd.get_dummies(test_X)
# # one_hot_imputed_test_X = my_imputer.fit_transform(one_hot_test_X)
# final_train, final_test = one_hot_train_X.align(one_hot_test_X, join='left',
#                                                        axis=1)
# final_imputed_train = my_imputer.fit_transform(final_train)
# final_imputed_test = my_imputer.transform(final_test)


# In[ ]:


# model = RandomForestRegressor()
# model.fit(final_imputed_train, target)
# preds = model.predict(final_imputed_test)
# print(preds)


# In[ ]:


submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)

