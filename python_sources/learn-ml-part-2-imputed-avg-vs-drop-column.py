#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Loading Code Runs At This Point
import pandas as pd
# Load data
file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
rawdata = pd.read_csv(file_path) 
print ('load data')
print (rawdata.shape)

# Choose target and predictors
# Use only columns that are supposed to have numbers (not words)
y = rawdata.SalePrice
# print(y.shape)
all_predictors = rawdata.drop(['SalePrice'], axis=1)
# print (all_predictors)
data = all_predictors.select_dtypes(exclude=['object'])
# print(data)
predictors = [col for col in data.columns]
# predictors = [col for col in data.columns 
#                                 if data[col].isnull().any()]
print ('only columns with numbers')
print(predictors)

# COUNT UP MISSING VALUES
missing_val_count_by_column = (data.isnull().sum())
print ('columns that are missing numbers or contain nan')
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# DROP COLUMN METHOD
# Identify columns of missing data, so they can also be removed from test set
cols_with_missing = [col for col in data.columns 
                                 if data[col].isnull().any()]
print ('removed columns that had nan values')
print(cols_with_missing)
filtered_data = data.drop(cols_with_missing, axis=1)
# END OF DROP COLUMN METHOD
print ('final dimensions of the data after column removal method')
print (filtered_data.shape)

# IMPUTATION METHOD
# Fill in missing data with column average
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
filtered_data = my_imputer.fit_transform(data)
# END OF IMPUTATION METHOD
print ('final dimensions of the data after imputation method')
print (filtered_data.shape)

# HAND SELECTED PREDICTORS
# predictors = ['TotRmsAbvGrd', 'FullBath', 'HalfBath', 'LotArea', 
#                         'YearBuilt']
# THIS DIDN'T WORK x = filtered_data[predictors]
# NEITHER DID THIS x = filtered_data.drop(predictors, axis=1)
x = filtered_data


# from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
# train_x, val_x, train_y, val_y = train_test_split(x, y,random_state = 2)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
forest_model = RandomForestRegressor()
forest_model.fit(x, y)
# melb_preds = forest_model.predict(val_x)
# print(mean_absolute_error(val_y, melb_preds))
# Read the test data
rawtest = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# remove columns that contain strings (as found in training data)
test = rawtest[predictors]
print ('test data dimensions')
print (test.shape)

# DROP COLUMN METHOD
# remove columns that had missing data in the training set
filtered_test = test.drop(cols_with_missing, axis=1)
# END OF DROP COLUMN METHOD
print ('test data dimensions after drop columns method')
print (filtered_test.shape)

# IMPUTATION METHOD
# Fill in missing data with column average
filtered_test = my_imputer.fit_transform(test)
# END OF IMPUTATION METHOD
print ('test data dimensions after imptations method')
print (filtered_test.shape)

# Treat the test data in the same way as training data. In this case, pull same columns.
# imputing function already had those columns removed
# test_x = filtered_test[predictors]
test_x = filtered_test
# Use the model to make predictions
predicted_prices = forest_model.predict(test_x)
# We will look at the predicted prices to ensure we have something sensible.
print ('predicted prices from the test set')
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# 

# 
