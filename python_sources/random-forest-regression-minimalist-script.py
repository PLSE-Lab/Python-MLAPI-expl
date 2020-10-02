#!/usr/bin/env python
# coding: utf-8

# This is a minimalist script which applies random forest regression from scikit-learn, to the 'House Prices' data set. It produces a score of around 0.17489, but this is not good, nor is it the point: the purpose of this script is to serve as a basic starting framework from which you can launch your own feature engineering.

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a minimal script to perform a regression on the kaggle 
# 'House Prices' data set using XGBoost Python API 
# Carl McBride Ellis (16.IV.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas  as pd

#===========================================================================
# read in the data from your local directory
#===========================================================================
train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ['OverallQual', 'GrLivArea', 'GarageCars',  'TotalBsmtSF']

#===========================================================================
#===========================================================================
X_train       = train_data[features]
y_train       = train_data["SalePrice"]
final_X_test  = test_data[features]

#===========================================================================
# essential preprocessing: imputation; substitute any 'NaN' with mean value
#===========================================================================
X_train      = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())

#===========================================================================
# perform the regression 
#===========================================================================
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, max_depth=7)
regressor.fit(X_train, y_train)

#===========================================================================
# use the model to predict the prices for the test data
#===========================================================================
predictions = regressor.predict(final_X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})
output.to_csv('submission.csv', index=False)

