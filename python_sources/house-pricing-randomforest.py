#!/usr/bin/env python
# coding: utf-8

# **Random Forest vs Linear Regression**
# 
# In this notebook, a comparison between **Random forest** algorithm and **Linear Regression** has been portrayed. Furthermore, Mean Absolute Error is calculated for both the approaches and it is clear that the random forest regressor performs better than the linear regression for the given feature set
# 
# The data is pre-processed to scale it and bring it in the range 0,1 and it is achieved via MinMaxScaler function. Also, the empty places in the data are filled with zeros there by bringing the data to a standard form.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train_df.head()


# In[ ]:


feature_selected = ['LotArea', 'YearBuilt', '1stFlrSF', 
                    '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 
                    'TotRmsAbvGrd', 'MSSubClass', 'OverallCond','KitchenQual']
train_df[feature_selected]


# In[ ]:


y_train = train_df.SalePrice

temp_train = train_df[feature_selected]


# In[ ]:


#get the dummies for text data
temp_train = pd.get_dummies(temp_train, columns=['KitchenQual'])
#fill empty spaces with zero
temp_train.fillna(value=0.0, inplace=True)
temp_train.head()


# In[ ]:


X_train = temp_train
#scaling the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train


# In[ ]:


#splitting data
state = 12  
test_size = 0.20  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=test_size, random_state=state)


# **RANDOM FOREST REGRESSOR**

# In[ ]:


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=12)
rf_model.fit(X_train, y_train)
rf_val_predictions = rf_model.predict(X_val)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_val)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# In[ ]:


#to predict on test data
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test_feature_selected = ['LotArea', 'YearBuilt', '1stFlrSF', 
                    '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 
                    'TotRmsAbvGrd', 'MSSubClass', 'OverallCond','KitchenQual']

X_test = test[test_feature_selected]
temp_test = X_test
X_test.head()


# In[ ]:


#get the dummies for text data
temp_test = pd.get_dummies(temp_test, columns=['KitchenQual'])
#fill empty spaces with zero
temp_test.fillna(value=0.0, inplace=True)

X_test = temp_test

X_test = scaler.fit_transform(X_test)
X_test


# **LINEAR REGRESSION**

# In[ ]:


#Initialize the Intercept and slope of linear regression
lnr_reg_model = LinearRegression().fit(X_train, y_train)
print('score on training data: ', lnr_reg_model.score(X_train,y_train))
print('co-efficient of X: ', lnr_reg_model.coef_)
print('Intercept of model: ', lnr_reg_model.intercept_)

#Predicting validation data
reg_model_pred = lnr_reg_model.predict(X_val)
mae_reg = mean_absolute_error(reg_model_pred, y_val)
print("Validation MAE for Linear Regression Model: {:,.0f}".format(mae_reg))
print('Score on validation: ', lnr_reg_model.score(X_val, y_val))


# In[ ]:


test_preds = rf_model.predict(X_test)
output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

