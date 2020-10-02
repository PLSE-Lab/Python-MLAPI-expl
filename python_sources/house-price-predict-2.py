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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

house_price_path = "../input/train.csv"

house_price = pd.read_csv(house_price_path)

house_price.describe()
house_price.head()
house_price.columns

y = house_price.SalePrice
feature_name = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    #'LotArea', 'LotShape', 'Utilities', 'Neighborhood', 'Condition1', 'HouseStyle', 'YearBuilt', 'BsmtCond', 'Heating', 'CentralAir', 'Fireplaces', 'GarageArea', 'OpenPorchSF', 'PoolArea']
X = house_price[feature_name]

pricing_model = DecisionTreeRegressor(random_state=1)
pricing_model.fit(X,y)

price_predict = pricing_model.predict(X)

mean_err = mean_absolute_error(y, price_predict)

print(mean_err)