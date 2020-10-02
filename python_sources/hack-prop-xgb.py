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
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('../input/train.csv', header=0)
print(train_df.shape)
print(train_df.head())

"""
train_df['hour'] = train_df['Date_Time'].apply(lambda x: x.split('_')[1][0:2])
train_df['hour'] = train_df.hour.astype(int)
train_df['month'] = train_df['Date_Time'].apply(lambda x: int(x.split('-')[1]))
train_df.dtypes
"""

xgb_reg = xgb.XGBRegressor(max_depth=10)
#X, y = train_df[train_df.columns.difference(['T', 'Date_Time', 'CO(GT)', 'NMHC(GT)', 'NO2(GT)', 'NOx(GT)'])], train_df['T']
X, y = train_df[train_df.columns.difference(['T', 'Date_Time'])], train_df['T']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
xgb_reg.fit(X_train, y_train)
preds = xgb_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
test_df = pd.read_csv('../input/test.csv')
"""
test_df['hour'] = test_df['Date_Time'].apply(lambda x: x.split('_')[1][0:2])
test_df['hour'] = test_df.hour.astype(int)
test_df['month'] = test_df['Date_Time'].apply(lambda x: int(x.split('-')[1]))
test_df.head()
"""
test_values = xgb_reg.predict(test_df[test_df.columns.difference(['Date_Time'])])
submission = pd.DataFrame(columns=['Date_Time', 'T'])
submission['Date_Time'] = test_df['Date_Time']
submission['T'] = test_values
submission.to_csv('submission.csv', index=False)


