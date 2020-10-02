#!/usr/bin/env python
# coding: utf-8

# This is *yet another* CatBoost scirpt, and yet another of my *minimalist* scripts. (I write them for my own amusement, but if they turn out to be useful to anybody, then all the better!) For feature selection I have used the top eight features obtained from my scikit-learn [recursive feature elimination](https://www.kaggle.com/carlmcbrideellis/recursive-feature-elimination-hp-v1) script. This script produces virtually the same score as that obtained in the aforementioned script, *i.e.* `0.153xx` (Note: that time I was using the scikit-learn Random Forest Regressor routine).
# ###### External link:
# * [CatBoost: a high-performance open source library for gradient boosting on decision trees](https://catboost.ai/)

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a minimal script to perform a regression on the kaggle 
# 'House Prices' data set using CatBoost
# Carl McBride Ellis (21.V.2020)
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
# my selection of features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]

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
from catboost import CatBoostRegressor
regressor = CatBoostRegressor(loss_function='RMSE', verbose=False)
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

