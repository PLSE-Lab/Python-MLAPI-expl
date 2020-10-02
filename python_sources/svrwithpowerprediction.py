# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
dataset = pd.read_csv('../input/AY_2SON1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 14:15].values

  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
a = r2_score(y_test, y_pred)
print(a)







