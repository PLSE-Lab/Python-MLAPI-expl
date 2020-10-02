import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

input_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

input_df.dropna(axis = 0 , subset = ['SalePrice'] , inplace = True)
target_SalePrice = input_df.SalePrice
input_predictors = input_df.drop(['SalePrice'] , axis = 1)
input_predictors_numeric = input_predictors.select_dtypes (exclude = ['object'])
test_predictors_numeric = test_df.select_dtypes (exclude = ['object'])

data_imputer = Imputer()
train_input_predictors_numeric = data_imputer.fit_transform(input_predictors_numeric)
test_predictors_numeric = data_imputer.transform(test_predictors_numeric)

Regressor_model = XGBRegressor()
Regressor_model.fit(train_input_predictors_numeric , target_SalePrice , verbose = False)
predicted_output = Regressor_model.predict(test_predictors_numeric)

XGBoost_Submission = pd.DataFrame({'Id': test_df.Id , 'SalePrice': predicted_output})
XGBoost_Submission.to_csv('XGBoost_Submission_File.csv' , index = False)