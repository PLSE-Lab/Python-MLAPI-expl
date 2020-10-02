# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as pl
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/Iris.csv")
dataset = dataset.drop('Id', 1)

# Split-out validation dataset
array = dataset.values
# In these lines we select which part of the matrix is input and which is output/class
X = array[:,1:4]
Y = array[:,4]
# The validation_size parameter will determine how much of samples will be used for train and test
validation_size = 0.30
# It's a good practice run a couple of time the script with different seeds
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Creates a model
knn = KNeighborsClassifier(n_neighbors=1)
# Train the model
knn.fit(X_train, Y_train)
# Score the classifier prediction
Y_predict = knn.predict(X_validation)
print("Accuracy: %s\n" % accuracy_score(Y_validation, Y_predict)) 
print("Confusion Matrix\n %s \n" % confusion_matrix(Y_validation, Y_predict, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]))
print("Classification Report\n %s \n" % classification_report(Y_validation, Y_predict, labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"]))

