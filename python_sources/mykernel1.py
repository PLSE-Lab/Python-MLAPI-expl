# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:50:15 2019

@author: kartik jaspal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' Data Preparation'''
dataset= pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")
x= dataset.iloc[:,:-1]
x=x.iloc[:,1:].values
y=dataset.iloc[:,7].values

'''Feature Selection , Final'''
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((414,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
"""regressor_ols=sm.OLS(endog=y,exog=x_opt).fit() 
regressor_ols.summary()"""

x_opt=x_opt[:,1:]


"""Spliting of Dataset""" 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x_opt,y,test_size=0.2,random_state=0)

"""Multiple Linear Regression Model""" 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

"""Testing Performance"""
regressor.score(x_test,y_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_error
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
