# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.ensemble import RandomForestRegressor

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head(n=5)
train.columns
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X,train_y)

test = pd.read_csv('../input/train.csv')
test_X = test[predictor_cols]
predicted_prices = my_model.predict(test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'ID':test.Id,'SalePrice':predicted_prices})
my_submission.to_csv('submission.csv',index=False)

