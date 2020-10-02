# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv(r"../input/Admission_Predict.csv", engine='python')

X = data.values[:,[0,1,2,3,4,5,6,7]]
y = data.values[:, [8]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
 
 
linereg = LinearRegression()
linereg.fit(X_train, y_train)
 
y_pred_class = linereg.predict(X_test)
  
print ('R^2 score:', metrics.r2_score(y_test, y_pred_class))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred_class))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred_class))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test,y_pred_class)))
  

