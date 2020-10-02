#!/usr/bin/env python
# coding: utf-8

# # Steps in this notebook:
# 1. Pre-processing:
#   Read the data from the input. Transform object types to categorical data. Handle missing values.
# 2. Try Decision Tree, Random Forest and XGBoost model with the training data.
# 3. Choose the model with best performance. In this case it's XGBoost Model. Predict home values in the test data with that model.

# In[ ]:


## Import necessary packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing.imputation import Imputer


# # Pre-processing
# ## Read the training data and test data from the input

# In[ ]:


iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)


# ## Use one-hot-encoding to get dummy variables

# In[ ]:


new_home_data = pd.get_dummies(home_data)
new_test_data = pd.get_dummies(test_data)


# ## Use imputation to handle missing values

# In[ ]:


missing_val_count_by_column = (new_home_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[ ]:


# Imputation
my_imputer = Imputer()
imputed_home_data = my_imputer.fit_transform(new_home_data)
imputed_test_data = my_imputer.fit_transform(new_test_data)


# ## The remaining things to do in pre-processing:
# 1. The type of data is numpy array. To change it into a pandas dataframe, use DataFrame constructor as belows.
# 2. Ensure the test data is encoded in the same manner as the training data with the align command.

# In[ ]:


train_df = pd.DataFrame(data=imputed_home_data, columns=new_home_data.columns) 
test_df = pd.DataFrame(data=imputed_test_data, columns=new_test_data.columns) 
final_train, final_test = train_df.align(test_df, join='inner', axis=1)


# # Model Training

# In[ ]:


# Create target object and call it y
y = home_data.SalePrice
X = final_train.drop(['Id'], axis=1)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[ ]:


# Model1: Decision Tree Regressor
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# **Result for Decision Tree Regressor:**
# * Validation MAE when not specifying max_leaf_nodes: 27,477
# * Validation MAE for best value of max_leaf_nodes: 24,373

# In[ ]:


# Model2: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# **Result for Randam Forest Regressor**
# * Validation MAE for Random Forest Model: 18,197

# In[ ]:


# Model3: XGB Regressor
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
xgb_model.fit(train_X, train_y, verbose=False)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for XGBoost Model before tuning: {:,.0f}".format(xgb_val_mae))

# Add tuning parameters: n_estimators, learning_rate and early_stopping_rounds
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05)
xgb_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], verbose=False)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for XGBoost Model after tuning: {:,.0f}".format(xgb_val_mae))


# **Result for XGBoost Regressor**
# * Validation MAE for XGBoost Model before tuning: 14,967
# * Validation MAE for XGBoost Model after tuning: 14,746

# # Creating a Model For the Competition
# 
# Build a Random Forest model and train it on all of **X** and **y**.  

# In[ ]:


# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = XGBRegressor(n_estimators=300, learning_rate=0.05)

# fit rf_model_on_full_data on all data from the 
rf_model_on_full_data.fit(X, y, verbose=False)


# # Make Predictions

# In[ ]:


# make predictions which we will submit. 
test_X = final_test.drop(['Id'], axis=1)
test_preds = rf_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

