# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
from sklearn.datasets import load_iris
iris_dataset  = load_iris()
# print("keys of iris dataset \n:{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR']+"\n...END")
#print("Target name :{}".format(iris_dataset['target_names']))
#print("Features  :{}".format(iris_dataset['feature_names']))
#print("Shape of Data : {}".format(iris_dataset['data'].shape))
#print("First five dataset : \n{}".format(iris_dataset['data'][:5]))
#print("target :{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print("X_train shape :{}".format(X_train.shape))
print("y_train shape :{}".format(y_train.shape))
print("X_test :{}".format(X_test.shape))
print("y_test :{}".format(y_test.shape))
iris_dataframe = pd.DataFrame(X_train,columns = iris_dataset.feature_names)
frr = pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(10,10),marker="o",hist_kwds={"bins":20},s=60,alpha=.8)
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors = 1)
Knn.fit(X_train,y_train)
X_new = np.array([[1.5,2.9,3,2.4]])
prediction = Knn.predict(X_new)
print("Shape of Prediction :{}".format(prediction.shape))
print("Predicted Flower : {}".format(iris_dataset['target_names'][prediction]))
#Note that we made the measurements of this single flower into a row in a twodimensional NumPy array, as scikit-learn always expects two-dimensional arrays
#for the data.
print("Shape of X_new : {}".format(X_new.shape))
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = Knn.predict(X_new)
print("Shape of Prediction :{}".format(prediction.shape))
print("Predicted Flower : {}".format(iris_dataset['target_names'][prediction]))
y_pred = Knn.predict(X_test)
y_pred
print("Accuracy : {}".format(np.mean(y_pred==y_test)))
print("Accuray by score method : {}".format(Knn.score(X_test,y_test)))

# Any results you write to the current directory are saved as output.