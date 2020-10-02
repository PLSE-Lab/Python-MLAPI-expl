# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dataset = pd.read_csv("../input/Admission_Predict.csv")
dataset=dataset.drop('Serial No.', 1)
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

#Splitting Training and Testing Data 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size = 0.2, random_state = 0)
                                                                            
#Fitting Multiple Linear Regression on Traing set###
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)  

#Prediting the test Results##
y_pred = regressor.predict(features_test)
y_pred_train = regressor.predict(features_train)

#Building Optimal model using Backward Propagation###
import statsmodels.formula.api as sm
features = np.append(arr = np.ones((400, 1)).astype(int), values = features, axis = 1)
#creating high impact features (removed SOP which is greater than P-value i.e. >0.05)
features_opt = features[:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
#Removed the predictor variable which higher p value(removed University rating)
features_opt = features[:, [0, 1, 2, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()
#All the other Independent variable P-value is less than 0.05 i.e. these variable influence the test results

