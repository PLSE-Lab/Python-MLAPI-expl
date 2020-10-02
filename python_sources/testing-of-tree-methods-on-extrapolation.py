# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# the following kernal is dedicated to test whether tree-based methods (such as random forest, XGboost can extropolate) on a simple linearly increasing trend

np.random.seed(0)
size = 5000
#A dataset with 3 features, with X increases with time (treat index as time)
X0 = np.random.normal(0, 1, (size, 1))
X0 = np.sort(X0, axis = 0)
X1 = np.random.normal(0, 1, (size, 1))
X1 = np.sort(X1, axis = 0)
X2 = np.random.normal(0, 1, (size, 1))
X2 = np.sort(X2, axis = 0)
X = np.concatenate((X0,X1,X2),axis = 1)
#Y = X0 + 2*X1 + noise
Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 0.5, size)
#normalize X and Y
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
# method 1: standardize data (so mean = 0, SD = 1)
X = scale(X)
# method 2: min-max normalization
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)
# define train and test
X_train_feature = X[0:4000,] 
X_test_feature = X[4000:,]
Y_train = Y[0:4000,]
Y_test = Y[4000:,]
y = Y_train

## Model 1: random forest regression with CV (accuracy = 87.4%, rmse = 0.13868797129524404)
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
model_rf = RandomForestRegressor(n_estimators = 1000, random_state = 42).fit(X_train_feature, y)
rf_pred = model_rf.predict(X_test_feature)

## Model 2: LASSO
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=0.0005).fit(X_train_feature, y)
lasso_pred = model_lasso.predict(X_test_feature)

## Model 3: Linear
from sklearn.linear_model import LinearRegression
model_linear = LinearRegression().fit(X_train_feature, y)
linear_pred = model_linear.predict(X_test_feature)

## Model 4: XGboost
import xgboost
# Step 3: XGBoost
depth = 2
rate = 0.07
estimators = 1200
g = 0
model_XGB = xgboost.XGBRegressor(max_depth = depth, n_estimators = estimators,learning_rate = rate).fit(X_train_feature, y)
XGB_pred = model_XGB.predict(X_test_feature)

## Comparison
## as shown on the graph, linear model (Lasso, Linear regression) can capture the trend in data extrapolate well, while RF and XGboost give "flat" prediction
import matplotlib.pyplot as plt
predictions = pd.DataFrame({"XGB":XGB_pred, "lasso":lasso_pred, "randomforest":rf_pred,"linear":linear_pred,
    "True":Y_test, "X1":X_test_feature[:,1]})
plt.scatter( 'True', 'lasso', data=predictions, marker='.', color='olive', linewidth=2)
plt.scatter( 'True', 'randomforest', data=predictions, marker='.', color='pink', linewidth=2)
plt.scatter( 'True', 'linear', data=predictions, marker='.', color='tomato', linewidth=2)
plt.scatter( 'True', 'XGB', data=predictions, marker='.', color='skyblue', linewidth=2)
plt.xlabel("True Value")
plt.ylabel("Predictions")
plt.legend()

plt.figure(3)
plt.scatter( 'X1', 'lasso', data=predictions, marker='.', color='olive', linewidth=2)
plt.scatter( 'X1', 'randomforest', data=predictions, marker='.', color='pink', linewidth=2)
plt.scatter( 'X1', 'linear', data=predictions, marker='.', color='tomato', linewidth=2)
plt.scatter( 'X1', 'XGB', data=predictions, marker='.', color='skyblue', linewidth=2)
plt.xlabel("X1")
plt.ylabel("Predictions")
plt.legend()