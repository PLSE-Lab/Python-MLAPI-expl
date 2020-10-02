# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

# read the data

df_train = pd.read_csv('../input/finec-1941-hw1/techparams_train.csv', index_col=0)
df_test = pd.read_csv('../input/finec-1941-hw1/techparams_test.csv', index_col=0)




x_train = df_train.drop('target', axis=1)
y_train = df_train['target']
x_test = df_test

# make OHE over full dataset
x_full = pd.concat([x_train, x_test], axis=0)

cols_to_encode = x_full.columns[x_full.nunique() <= 40]
dummies = [pd.get_dummies(x_full[c]) for c in cols_to_encode]
x_full = pd.concat([x_full.drop(cols_to_encode, axis=1)] + dummies, axis=1)

x_train = x_full.iloc[:len(x_train)]
x_test = x_full.iloc[len(x_train):]
x_train.shape, y_train.shape, x_test.shape

# fit model
lr = LinearRegression()
lr.fit(x_train, y_train)
print('train mse:', mean_squared_error(y_train, lr.predict(x_train)))
y_test_baseline = pd.Series(data=lr.predict(x_test), index=x_test.index, name='target')
y_test_baseline.to_frame().to_csv('techparams_test_baseline.csv')