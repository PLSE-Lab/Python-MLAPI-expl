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
#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("../input/bike-sharing-dataset/day.csv")
X = dataset.iloc[:,2:13].values #Independent variables. Removed the instant and the dteday columns 
y = dataset.iloc[:,15].values #Dependent variable. The data is a range of numbers. Hence it is a regression problem

# Data preprocessing

#Categorical data. We have to OneHotEncode them

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = range(0,7)) #These are the columns containing categorical data
X = ohe.fit_transform(X).toarray()

#Splitting the data set into training and test set.
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0 )

#Random Forest Regression model

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 250, max_features = 'sqrt') #Chose the max_features value from the grid serach method
regressor.fit(X_train,y_train)

#Predicting the values 

y_pred = regressor.predict(X_test) #1-D Vector containing the predicted values of the dependent variable. That is total count of bike rented.

#Evaluating the model using k-fold cross validation technique

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10)
accuracy.mean()
accuracy.std()

# Grid search 

"""from sklearn.model_selection import GridSearchCV
parameters = [ {'n_estimators' :[200,220,250,300,325,350], 'max_features' : ['auto','sqrt' , 'log2'], 'min_samples_split' : [2,3,4,5,6] },
                ]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
"""