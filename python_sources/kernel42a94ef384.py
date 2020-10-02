# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/Admission_Predict.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
x=x[:,[0,1,2,3,4,6,7]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300)
regressor.fit(x_train,y_train)
'''import statsmodels.formula.api as sm
x=np.append(arr=np.ones((400,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,6,7]]
r_ols=sm.OLS(endog=y,exog=x_opt).fit()
r_ols.summary()'''


