#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:27:02 2018

@author: abinash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("../input/kc_house_data.csv")
mean = dataset.sqft_lot.mean()
std = dataset.sqft_lot.std()

dataset = dataset[dataset.loc[:, 'sqft_lot']>(mean-2*std)]
dataset = dataset[dataset.loc[:, 'sqft_lot']<(mean+2*std)]

X = dataset.iloc[:, 3:21].values
y = dataset.iloc[:, 2].values



X_new = X[:, [1,2,3,4,7,8,9,11,12,14,16,17]]


#Splitting the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.3,random_state = 0)

#making regression model
from sklearn.ensemble import RandomForestRegressor
regresor = RandomForestRegressor(n_estimators = 500)
regresor.fit(X_train, y_train)

#predicting values
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
try:
    y_pred = regresor.predict(X_test)
except NotFittedError as e:
    print(repr(e))
print((mean_absolute_error(y_test, y_pred)/np.mean(y_test))*100)