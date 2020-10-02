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


import pandas as pd
data = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(data)
print(iowa_data.describe())


# In[ ]:


print(iowa_data.columns)


# In[ ]:


iowa_data_price = iowa_data.SalePrice
print(iowa_data_price.head())


# In[ ]:


columns = ['GarageYrBlt' , 'PoolArea']
iowa_columns = iowa_data[columns]
iowa_columns.describe()


# In[ ]:


y = iowa_data.SalePrice
predictor = ['LotArea' ,'YearBuilt' , '1stFlrSF' , '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = iowa_data[predictor]
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X ,y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))


# In[ ]:


from sklearn.metrics import mean_absolute_error
predicted_prices = iowa_model.predict(X)
mean_absolute_error(y, predicted_prices)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
iowa_model = DecisionTreeRegressor()
iowa_model.fit(train_X, train_y)
validated_prediction = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, validated_prediction))


# In[ ]:


def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
iowa_predix = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_predix))


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


# In[ ]:


test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_X = test[predictor_cols]
predicted_prices = my_model.predict(test_X)
print(predicted_prices)


# In[ ]:


my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)

