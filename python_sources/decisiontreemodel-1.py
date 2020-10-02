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


train_file_path = '../input/home-data-for-ml-course/train.csv'


# In[ ]:


train_data = pd.read_csv(train_file_path)


# In[ ]:


train_data.columns


# In[ ]:


train_data.describe()


# In[ ]:


train_data.head()


# In[ ]:


train_data.columns


# In[ ]:


y = train_data.SalePrice


# In[ ]:


X = X.select_dtypes(exclude=['object'])


# In[ ]:


X.describe()


# In[ ]:


X.columns


# In[ ]:


# Test Data Missing Values

# (1459, 33)
# BsmtFinSF1      1
# BsmtFinSF2      1
# BsmtUnfSF       1
# TotalBsmtSF     1
# BsmtFullBath    2
# BsmtHalfBath    2
# GarageCars      1
# GarageArea      1
# dtype: int64


# In[ ]:


print(X.shape)


# In[ ]:


train_data_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']


# In[ ]:


X = train_data[train_data_features]


# In[ ]:


X.head()


# In[ ]:


X.describe()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
train_data = DecisionTreeRegressor()


# In[ ]:


s = (X.dtypes == 'object')
object_cols = list(s[s].index)
object_cols


# In[ ]:


print("Categorical variables:")
print(object_cols)


# In[ ]:


X = X.select_dtypes(exclude=['object'])
X


# In[ ]:


X.head()


# In[ ]:


X.describe()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


train_model = DecisionTreeRegressor(random_state=1)


# In[ ]:


train_model.fit(X,y)


# In[ ]:


print("Making predictions for the following 5 houses:")
print(X.head())


# In[ ]:


print("The predictions are")
print(train_model.predict(X.head()))


# In[ ]:


train_model.predict(X)


# In[ ]:


y


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


predicted_train_model = train_model.predict(X)


# In[ ]:


mean_absolute_error(predicted_train_model,y)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)


# In[ ]:


tain_model = DecisionTreeRegressor(random_state=1)


# In[ ]:


train_model.fit(train_X,train_y)


# In[ ]:


val_predictions = train_model.predict(val_X)


# In[ ]:


val_mae = mean_absolute_error(val_predictions, val_y)


# In[ ]:


print('Validation MAE: {:,.0f}'.format(val_mae))


# In[ ]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[ ]:


canditate_max_leaf_nodes = [5, 10, 25, 35, 50, 65, 80, 100, 250, 500, 1000]


# In[ ]:


# A loop to find the ideal tree size from canditate_max_leaf_nodes
for max_leaf_nodes in canditate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, val_X, train_X, val_y, train_y)
    print('mae: ',  max_leaf_nodes, my_mae) 


# In[ ]:


final_model = DecisionTreeRegressor(max_leaf_nodes=50, random_state=1)


# In[ ]:


print(final_model)


# In[ ]:


final_model.fit(X,y)


# In[ ]:


y


# In[ ]:


print(final_model)


# In[ ]:


y


# In[ ]:


X


# In[ ]:


y


# In[ ]:


output = pd.DataFrame({'Id': train_data.Id,
                       'SalePrice': y})
output.to_csv('submission.csv', index=False)


# In[ ]:


test_file_path = '../input/home-data-for-ml-course/test.csv'


# In[ ]:


test_data = pd.read_csv(test_file_path)


# In[ ]:


test_data.describe()


# In[ ]:


test_data


# In[ ]:


test_X = test_data[train_data_features]


# In[ ]:


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_test_X = pd.DataFrame(my_imputer.fit_transform(test_X))
imputed_test_X.columns = test_X.columns


# In[ ]:




print(imputed_test_X)

missing_val_count_by_column = (imputed_test_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


predict_test = final_model.predict(imputed_test_X)


# In[ ]:


print(predict_test)


# In[ ]:


my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predict_test}) 
my_submission.to_csv('submission.csv', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




