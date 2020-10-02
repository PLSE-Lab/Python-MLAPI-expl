"""
@script-author: Sarvesh...
@script-name: KNN algorithm using dataset.
@script-description:For the given test data using KNN algorithm predicts spicies of flower, by learning from Train dataset.
@external-package-used:numpy.pandas,scikit-learn(sklearn).

"""

 # linear algebra
import numpy as np
 # data processing
import pandas as pd
 # the KNN from Scikit-lean
from sklearn.neighbors import KNeighborsClassifier
 # For splitting the data into train and test set
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
import os
print(os.listdir("../input"))
filename="../input/iris/Iris.csv"
iris=pd.read_csv(filename)
#allow us to see all columns and rows:
pd.set_option('display.max_columns', None)#by this we can view all the data according to the screen size
pd.set_option('display.max_rows', None)
print(iris.head(n=5))#A simple head function 
print("-"*80)
print(iris.tail(n=5))#A simple tail function
print("-"*80)
print(iris.keys())

# For the sake of accuracy of KNN model, we need to randomly split the data 
knn=KNeighborsClassifier(n_neighbors=11)
#To convert Species into dummy variables ( 0 and 1 here setosa is 0 and versicolor is 1)
iris_dum=pd.get_dummies(iris)
print(iris_dum.head())
print(iris_dum.tail())
y=iris_dum['Species_Iris-virginica'].values
X=iris_dum.drop('Species_Iris-virginica', axis=1).values
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.7)
#Fitting the Classifier into train data
knn.fit(X_train,y_train)

# For Prediction the labels for training data
y_predict=knn.predict(X_test)

# This .score() method the accuracy of classifiers prediction can be computed and printed
print("score=",knn.score(X_test, y_test))