# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor

# Path of file to read
train_file_path = '../input/train.csv'

# Home Data
home_data = pd.read_csv(train_file_path)

# X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd','YearBuilt', 'GrLivArea',
'OverallQual', 'OverallCond','FullBath','KitchenAbvGr']

train_X = home_data[features]
# y
train_y = home_data.SalePrice

# Create training predictors data
my_model = RandomForestRegressor(random_state=1)
my_model.fit(train_X, train_y)

# Test
test_file_path = '../input/test.csv'
test = pd.read_csv(test_file_path)

test_X = test[features]

# Model to make predictions
predicted_prices = my_model.predict(test_X)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

