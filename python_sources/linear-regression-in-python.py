# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Importing the dataset
Practice_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train =  Practice_data.dropna()
test = test_data.dropna()

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1:].values
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1:].values

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

plt.title('Relationship between Testing sets')
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.show()
