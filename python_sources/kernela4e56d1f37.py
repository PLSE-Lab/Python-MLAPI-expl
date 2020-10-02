# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

X_feat = ['YearBuilt', 'LotFrontage', 'LotArea', 'HouseStyle', 'Heating', 'BedroomAbvGr', 'KitchenAbvGr', 'SaleType']
y = data.SalePrice
X = data[X_feat]
X_test = data_test[X_feat]
one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='left',axis=1)


d_model = make_pipeline(Imputer(), DecisionTreeRegressor(max_leaf_nodes=18,random_state=1))
r_model = make_pipeline(Imputer(), RandomForestRegressor(random_state=1))

train_X, val_X, train_y, val_y = train_test_split(final_train,y,random_state=1)


d_model.fit(train_X,train_y)
r_model.fit(train_X,train_y)

d_pre = d_model.predict(val_X)
r_pre = r_model.predict(val_X)

print(mean_absolute_error(d_pre,val_y))
print(mean_absolute_error(r_pre,val_y))

test_pre = r_model.predict(final_test)

output = pd.DataFrame({'Id': data_test.Id,
                       'SalePrice': test_pre})

output.to_csv('submission.csv', index=False)

