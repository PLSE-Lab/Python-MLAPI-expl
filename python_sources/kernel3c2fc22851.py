import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

read_train_data = pd.read_csv('../input/traindata/train.csv')
read_test_data = pd.read_csv('../input/testdata/test.csv')

train_SalePrice = read_train_data.SalePrice
train_predictors = read_train_data.drop(['SalePrice'] , axis = 1)
train_predictors_numeric = train_predictors.select_dtypes(exclude = ['object'])

test_predictors_numeric = read_test_data.select_dtypes(exclude = ['object'])

data_imputer = Imputer()
train_predictors_numeric_imputed = data_imputer.fit_transform(train_predictors_numeric)

test_predictors_numeric_imputed = data_imputer.transform(test_predictors_numeric)

predict_model = RandomForestRegressor()
predict_model.fit(train_predictors_numeric_imputed , train_SalePrice)
predict_output = predict_model.predict(test_predictors_numeric_imputed)

submission_file = pd.DataFrame({'Id' : read_test_data.Id , 'SalePrice' : predict_output})
submission_file.to_csv('Submission_file_03012018.csv' , index = False)