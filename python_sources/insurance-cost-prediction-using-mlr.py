# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 00:17:52 2018

@author: Kenil
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv('../input/insurance.csv')
x = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,6].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,1] =labelencoder_x.fit_transform(x[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
x[:,4] =labelencoder_x.fit_transform(x[:,4])
onehotencoder = OneHotEncoder(categorical_features = [4])
x[:,5] =labelencoder_x.fit_transform(x[:,5])
onehotencoder = OneHotEncoder(categorical_features = [5])
x = onehotencoder.fit_transform(x).toarray()

x = x[:,1:]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
