# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data ploting, CSV file

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Importing Dataset csv
dataset = pd.read_csv('../input/Salary_Data.csv')

#checking for NaN values
dataset.isnull().sum()

#splitting the dataset - dependent and independent variables
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set
y_pred=regressor.predict(x_test)

#Visualising the training set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Experience vs Salary Plot')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Visualisation of the trest sest set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Experience vs Salary plot')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()