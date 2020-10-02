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

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from math import log

## Data input
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
complete_data = train_data.append(test_data)

##Cleaning up NA items in data
#Replacing Electrical NaN with the mode "SBrkr"
complete_data.loc[1379, 'Electrical'] = 'SBrkr'
#Replacing all other NA's with 0's
complete_data = complete_data.fillna(0)

 
## Creating dummies for regression
# Listing continuous and dummy variables
cont_vars = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
             'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal', 'YearRemodAdd'
              , 'YearBuilt','OverallCond', 'OverallQual','GarageYrBlt', 'SalePrice', 'Id']
dummy_vars = list(set(complete_data.columns) - set(cont_vars))
complete_data = pd.get_dummies(complete_data, columns = dummy_vars)

## Feature Engineering
complete_data['LotFrontage'] = np.log(complete_data['LotFrontage'])
complete_data['LotArea'] = np.log(complete_data['LotArea'])
complete_data['BsmtFinSF1'] = np.log(complete_data['BsmtFinSF1'])
complete_data['BsmtFinSF2'] = np.log(complete_data['BsmtFinSF2'])
complete_data['TotalBsmtSF'] = (complete_data['TotalBsmtSF'])**2
complete_data['1stFlrSF'] = np.log(complete_data['1stFlrSF'])
complete_data['2ndFlrSF'] = (complete_data['2ndFlrSF'])**2
complete_data['LowQualFinSF'] = np.log(complete_data['LowQualFinSF'])
complete_data['GrLivArea'] = np.log(complete_data['GrLivArea'])
complete_data['GrLivArea'] = (complete_data['GrLivArea'])**2
complete_data['GarageArea'] = np.log(complete_data['GarageArea'])
complete_data['WoodDeckSF'] = np.log(complete_data['WoodDeckSF'])
complete_data['OpenPorchSF'] = np.log(complete_data['OpenPorchSF'])
complete_data['EnclosedPorch'] = np.log(complete_data['EnclosedPorch'])
complete_data['3SsnPorch'] = np.log(complete_data['3SsnPorch'])
complete_data['ScreenPorch'] = np.log(complete_data['ScreenPorch'])
complete_data['PoolArea'] = np.log(complete_data['PoolArea'])
complete_data['MiscVal'] = np.log(complete_data['MiscVal'])
complete_data['OverallCond'] = np.log(complete_data['OverallCond'])
complete_data['OverallQual'] = (complete_data['OverallQual'])**2
complete_data['GarageYrBlt'] = np.log(complete_data['GarageYrBlt'])
complete_data['SalePrice'] = np.log(complete_data['SalePrice'])
complete_data['LotFrontage-x-LotArea'] = complete_data['LotFrontage'] * complete_data['LotArea']
complete_data['LotArea-x-GrLivArea'] = complete_data['LotArea'] * complete_data['GrLivArea']
complete_data['LotArea-x-GrLivArea-x-LotFrontage'] = complete_data['LotArea'] * complete_data['GrLivArea'] * complete_data['LotFrontage'] 
complete_data['OverallQual-x-OverallCond'] = complete_data['OverallQual'] * complete_data['OverallCond']
complete_data['Basement_dummy'] = complete_data['TotalBsmtSF'] > 0
complete_data['Garage_dummy'] = complete_data['GarageArea']>0
complete_data['Pool_dummy'] = complete_data['PoolArea']>0
complete_data['YearRemodAdd_dummy'] = complete_data['YearRemodAdd'] > complete_data['YearBuilt']
complete_data = complete_data.replace(-inf, 0)

## Splitting train and test data
train_data_x = (complete_data.loc[complete_data.Id < 1461]).drop(['SalePrice'], axis = 1)
train_data_y = (complete_data.loc[complete_data.Id < 1461]).SalePrice
submission_test_data_x = (complete_data.loc[complete_data.Id >= 1461]).drop(['SalePrice'], axis = 1)

## Splitting 70-30 for training the model
X_train,X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size = 0.30, 
                                                    random_state = 541)


## Lasso Regression (L1)) - Grid Search for optimal alpha(penalty)
#lasso_housing = Lasso()
#alpha_space = np.arange(0,20, .1)
#param_grid = {'alpha': alpha_space}
#lasso_cv = GridSearchCV(lasso_housing, param_grid, cv = 5)
#lasso_cv.fit(X_train, y_train)
#lasso_mape = np.mean(np.abs((y_test - lasso_cv.predict(X_test)) / y_test)) * 100
#print("Lasso MAPE:{}".format(lasso_mape))
#print("Lasso Alpha:{}".format(lasso_cv.best_params_))

## Ridge Regression (L2) - Grid Search for optimal alpha(penalty)
#ridge_housing = Ridge()
#alpha_space = np.arange(0,10, 0.1)
#param_grid = {'alpha': alpha_space}
#ridge_cv = GridSearchCV(ridge_housing, param_grid, cv = 5)
#ridge_cv.fit(X_train, y_train)
#ridge_mape = np.mean(np.abs((y_test - ridge_cv.predict(X_test)) / y_test)) * 100
#print("Ridge:{}".format(ridge_mape))
#print("Ridge Alpha:{}".format(ridge_cv.best_params_))


## Final Model for lasso based on alpha from grid search
final_model_lasso = Lasso(alpha = 0)
final_model_lasso_cvs = cross_val_score(final_model_lasso, X_train, y_train, cv = 5)
final_model_lasso.fit(X_train, y_train)

# Cross validation results using 5 fold
print("Lasso CV: {}".format(final_model_lasso_cvs))

## Final Model for Ridge based on alpha from grid search
final_model_ridge = Ridge(alpha = 7.6)
final_model_ridge_cvs = cross_val_score(final_model_ridge, X_train, y_train, cv = 5)
final_model_ridge.fit(X_train, y_train)

## Splitting 100-0 now once grid search and cross validation scores are known
X_train,X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size = 0.0001, 
                                                    random_state = 521084)

final_model_ridge=Ridge(alpha=7.6)
final_model_ridge.fit(X_train, y_train)

# Cross validation results using 5 fold
print("Ridge CV: {}".format(final_model_ridge_cvs))


## Submission - Choosing Ridge because of better Cross Validation Scores
predictions = {'Id':submission_test_data_x.Id, 'SalePrice': np.exp(final_model_ridge.predict(submission_test_data_x))}
predictions = pd.DataFrame(data = predictions)
predictions.to_csv("submission.csv", index = False)