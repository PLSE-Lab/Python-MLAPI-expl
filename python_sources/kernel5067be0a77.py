# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


'''
    @script author: Divya BM
    @script-name: KNN algorithm
    @script-description: KNN algorithm is used to fit a model and predit the target values using the test data.
    @external packages used: sklearn
    
'''
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.model_selection as ms
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#load dataset
iris = datasets.load_iris()
data=iris
#print the target and feature column names
a=iris.feature_names
b=iris.target_names
#print the dimension of the iris dataset
dim=iris.data.shape
#assigning X and Y values to fit a model
X = iris.data
Y = iris.target
#splitting the dataset into train and test data
x_train,x_test,y_train,y_test = ms.train_test_split(X,Y,test_size = 0.3, random_state = 192635)
dm = x_train.shape,x_test.shape,y_train.shape,y_test.shape 
#using classifier to use Knn algorithm
classifier = KNeighborsClassifier(n_neighbors = 5 )
classifier.fit(x_train,y_train)
#predicting the target values using the test dataset
y_pred = classifier.predict(x_test)
#print(classification_report(y_test, y_pred))
#error = []
#for i in range(1,40):
 #  knn = KNeighborsClassifier(n_neighbors = i)
 #  knn.fit(x_train, y_train)
 #  pred_i = knn.predict(x_test)
 #  error.append(np.mean(pred_i != y_test))####
#print(error)

#printing the test and predicted values
print("The test values : " + str(list(y_test)))
print("The predicted values:" + str(list(y_pred)))