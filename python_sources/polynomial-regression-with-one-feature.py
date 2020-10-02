#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 07:29:29 2018

@author: Pawanvir Singh

This is minimal Example of Polynomial 
Regression with One Variable 

as we Know when we add polynomial terms to 
Our regression hypothesis  the function will no more linear 
so it will lead to better fit our model to non linear data set
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
#loading the dataset
dataset= pd.read_csv("../input/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

#adding Polynomial for better fitting of data 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree= 2)
poly_features = poly.fit_transform(X)
poly.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(poly_features,y)
#normal regression
regressor=LinearRegression()
regressor.fit(X,y)
#ploting the data  
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 2 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()
#plotting linear hypothesis
plt.scatter(X,y)
plt.plot(X,regressor.predict(X))
plt.title("Linear  Regression Experiance Vs Salary  ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show() 
# higher degree
# Adding Polynominals to the hypothesis 
poly = PolynomialFeatures(degree= 3)
poly_features = poly.fit_transform(X)
poly.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(poly_features,y)
#ploting the data  for polynomial regression 
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 3 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()
#ploting the Linear regresson
plt.scatter(X,y)
plt.plot(X,regressor.predict(X))
plt.title("Linear  Regression Experiance Vs Salary  ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show() 

