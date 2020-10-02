# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model

train = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/train.csv')
X_train = train.iloc[:, 3:5]
y_train = train.iloc[:, 2:3]
for i in range(0, X_train.shape[0]):
    if X_train['effectiveness'][i] == 'Highly Effective':
        X_train['effectiveness'][i] = 0
    elif  X_train['effectiveness'][i] == 'Considerably Effective':
        X_train['effectiveness'][i] = 1
    elif  X_train['effectiveness'][i] == 'Moderately Effective':
        X_train['effectiveness'][i] = 2
    elif  X_train['effectiveness'][i] == 'Marginally Effective':
        X_train['effectiveness'][i] = 3
    else:
        X_train['effectiveness'][i] = 4

for i in range(0, X_train.shape[0]):
    if X_train['sideEffects'][i] == 'Extremely Severe Side Effects':
        X_train['sideEffects'][i] = 0
    elif  X_train['sideEffects'][i] == 'Severe Side Effects':
        X_train['sideEffects'][i] = 1
    elif  X_train['sideEffects'][i] == 'Moderate Side Effects':
        X_train['sideEffects'][i] = 2
    elif  X_train['sideEffects'][i] == 'Mild Side Effects':
        X_train['sideEffects'][i] = 3
    else:
        X_train['sideEffects'][i] = 4

valid = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/valid.csv')
X_valid = valid.iloc[:, 3:5]
y_valid = valid.iloc[:, 2:3]
for i in range(0, X_valid.shape[0]):
    if X_valid['effectiveness'][i] == 'Highly Effective':
        X_valid['effectiveness'][i] = 0
    elif  X_valid['effectiveness'][i] == 'Considerably Effective':
        X_valid['effectiveness'][i] = 1
    elif  X_valid['effectiveness'][i] == 'Moderately Effective':
        X_valid['effectiveness'][i] = 2
    elif  X_valid['effectiveness'][i] == 'Marginally Effective':
        X_valid['effectiveness'][i] = 3
    else:
        X_valid['effectiveness'][i] = 4

for i in range(0, X_valid.shape[0]):
    if X_valid['sideEffects'][i] == 'Extremely Severe Side Effects':
        X_valid['sideEffects'][i] = 0
    elif  X_valid['sideEffects'][i] == 'Severe Side Effects':
        X_valid['sideEffects'][i] = 1
    elif  X_valid['sideEffects'][i] == 'Moderate Side Effects':
        X_valid['sideEffects'][i] = 2
    elif  X_valid['sideEffects'][i] == 'Mild Side Effects':
        X_valid['sideEffects'][i] = 3
    else:
        X_valid['sideEffects'][i] = 4

test = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/test.csv')
X_test = test.iloc[:, 2:4]
for i in range(0, X_test.shape[0]):
    if X_test['effectiveness'][i] == 'Highly Effective':
        X_test['effectiveness'][i] = 0
    elif  X_test['effectiveness'][i] == 'Considerably Effective':
        X_test['effectiveness'][i] = 1
    elif  X_test['effectiveness'][i] == 'Moderately Effective':
        X_test['effectiveness'][i] = 2
    elif  X_test['effectiveness'][i] == 'Marginally Effective':
        X_test['effectiveness'][i] = 3
    else:
        X_test['effectiveness'][i] = 4

for i in range(0, X_test.shape[0]):
    if X_test['sideEffects'][i] == 'Extremely Severe Side Effects':
        X_test['sideEffects'][i] = 0
    elif  X_test['sideEffects'][i] == 'Severe Side Effects':
        X_test['sideEffects'][i] = 1
    elif  X_test['sideEffects'][i] == 'Moderate Side Effects':
        X_test['sideEffects'][i] = 2
    elif  X_test['sideEffects'][i] == 'Mild Side Effects':
        X_test['sideEffects'][i] = 3
    else:
        X_test['sideEffects'][i] = 4
X_train = np.vstack((X_train, X_valid))
y_train = np.vstack((y_train, y_valid))
reg_gradientboost = GradientBoostingRegressor(loss ='ls', n_estimators = 150)
reg_gradientboost.fit(X_train, y_train)
reg_knn = KNeighborsRegressor(n_neighbors=5)
reg_knn.fit(X_train, y_train)
reg_lasso = linear_model.Lasso(alpha=0.1)
reg_lasso.fit(X_train, y_train)

result_gradientboosting = reg_gradientboost.predict(X_test)
result_gradientboosting = result_gradientboosting.reshape(-1, 1)
for i in range(0, result_gradientboosting.shape[0]):
    if result_gradientboosting[i] < 1:
        result_gradientboosting[i] = 1
df_gradientboosting = pd.DataFrame(result_gradientboosting)

result_knn = reg_knn.predict(X_test)
result_knn = result_gradientboosting.reshape(-1, 1)
for i in range(0, result_knn.shape[0]):
    if result_knn[i] < 1:
        result_knn[i] = 1
df_knn = pd.DataFrame(result_knn)

result_lasso = reg_lasso.predict(X_test)
result_lasso = result_lasso.reshape(-1, 1)
for i in range(0, result_lasso.shape[0]):
    if result_lasso[i] < 1:
        result_lasso[i] = 1
df_lasso= pd.DataFrame(result_lasso)

weighted_result = []
for i in range(0, df_gradientboosting.shape[0]):
    weighted_result.append(df_gradientboosting[0][i]*0.4 + df_knn[0][i]*0.3 + df_lasso[0][i]*0.3)
weighted_result = np.array(weighted_result)
weighted_result = weighted_result.reshape(-1, 1)

submission = pd.read_csv('/kaggle/input/cdal-competition-2019-fall/submission.csv')
df_weighted = pd.DataFrame(weighted_result)
submission.rating = np.array(df_weighted).reshape(-1,1)
submission.to_csv('submission.csv', index=False)
