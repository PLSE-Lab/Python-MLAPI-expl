# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
    @script_author: Shekhina Neha
    @script_name: Flower species prediction using K-Nearest Neighbors (KNN) Algorithm.
    @script_description: The program predicts the species of flower for the Iris dataset.
    @script_package_used: sklearn
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# importing the necessary packages
import pandas as pd # data processing, CSV file I/O
from sklearn.neighbors import KNeighborsClassifier # KNN package
import sklearn.metrics as sm # accuracy
import sklearn.model_selection as ms # train-test split
# Reading the csv file
data = pd.read_csv('../input/iris/Iris.csv') 
# Checking for missing values
data.isnull().sum().sum()
# Removing the 'Id' column from the dataset
data1 = data.drop(['Id'], axis = 1)
# Obtain the dataset without the target variable
X = data1.drop(['Species'], axis = 1)
# Target data
Y = data1['Species']
# Splitting into train and test data
x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size = 0.3, random_state = 4)
# Finding the dimensions of the train and test data
x_train.shape, x_test.shape, y_train.shape, y_test.shape
# Creating a KNN Classifier object
neigh = KNeighborsClassifier(n_neighbors=5)
# Fitting the model to the training set
neigh.fit(x_train, y_train)
# Predicting the target values for the test data
y_pred = neigh.predict(x_test)
print("The predicted values are: ")
# Converting the array into dataframe
pred = pd.DataFrame(y_pred)
print(pred)
# Obtain the accuracy of the model
score = sm.accuracy_score(y_test, y_pred)
score

# %% [code]
