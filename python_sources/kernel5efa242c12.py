# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Import the dataset...
from sklearn.datasets import load_iris
data = load_iris()

#identify and assign the target variable...
X = data.data
Y = data.target

#Split the data into train and test...
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)

#Find the ideal k_value...
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
scores = []
for k in range(1,25,1):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,Y_train)
    Y_Pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(Y_test,Y_Pred))
    
ideal_k_value = scores.index(max(scores))

#Find the predicted values for the ideal_k_value
knn = KNeighborsClassifier(n_neighbors = ideal_k_value)
knn.fit(X_train,Y_train)
Y_Pred = knn.predict(X_test)

#output...
print("The accuracy for the the k value:",ideal_k_value," is ",max(scores))