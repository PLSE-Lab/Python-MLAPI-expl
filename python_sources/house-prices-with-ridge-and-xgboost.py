import numpy as np
import pandas as pd
import scipy as sp
import os

from sklearn import linear_model
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv("../input/train.csv")
data = data.fillna(0)

data_test = pd.read_csv("../input/test.csv")
data_test = data_test.fillna(0)


y = data['SalePrice']
data = data.drop(columns=['SalePrice'])

x = pd.get_dummies(data[data.columns[1:80]])
x /= x.max()

test = pd.get_dummies(data_test[data_test.columns[1:80]])
test /= test.max()

missing_cols = set( x.columns ) - set( test.columns )
for col in missing_cols:
    test[col] = 0
test = test[x.columns]

x['SalePrice'] = y
num_error = x.corr()['SalePrice'][:-1]
feature_list = num_error[abs(num_error) > 0.28].sort_values(ascending=False)

x = x[feature_list.index]
test = test[feature_list.index]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)

xgbr = xgb.XGBRegressor(objective="reg:linear", booster="gbtree", eta=0.05, gamma=0, max_depth=3, min_child_weight=4, subsample=1, colsample_bytree=1)
regr = linear_model.Ridge(normalize=True)

regr.fit(x, y)
xgbr.fit(x, y)
#regr.fit(x_train, y_train)
#xgbr.fit(x_train, y_train)

test_pred_xgb = xgbr.predict(test)
test_pred = regr.predict(test)
#y_pred_xgb = xgbr.predict(x_test)
#y_pred = regr.predict(x_test)

#y_pred = (y_pred_xgb + 2*y_pred) / 3
test_pred = (test_pred_xgb + 2*test_pred) / 3

#print('Variance score: %.5f' % r2_score(y_test, y_pred)) #0.79606 single model

output = pd.DataFrame()
output['Id'] = data_test['Id']
output['SalePrice'] = test_pred
output.to_csv('output.csv', index = False)