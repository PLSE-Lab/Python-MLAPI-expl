"""
@script-author: Ayush
@script-name: Predictionof Flower Spicies using KNN algorithm 
@script-description:For the given test data the algo predicts spicies of flower
@external-package-used:numpy.pandas,sklearn

"""


import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.neighbors import KNeighborsClassifier # the KNN from Scikit-lean
from sklearn.model_selection import train_test_split # this allows me to separate data into train and test 
from sklearn.metrics import mean_squared_error
import os
print(os.listdir("../input"))
filename="../input/iris/Iris.csv"
iris=pd.read_csv(filename)
#allow us to see all columns and rows:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(iris.head(n=5))
print("-"*80)
print(iris.tail(n=5))
print("-"*80)
print(iris.keys())

# we need to split the data randomly into test data set to show accuracy of model and train data set, 
knn=KNeighborsClassifier(n_neighbors=11)
# we need to convert Species into dummy variables ( 0 and 1 here setosa is 0 and versicolor is 1)
iris_dum=pd.get_dummies(iris)
print(iris_dum.head())
print(iris_dum.tail())
y=iris_dum['Species_Iris-virginica'].values
X=iris_dum.drop('Species_Iris-virginica', axis=1).values
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.7)
#fit the classifier to the train data
knn.fit(X_train,y_train)

# Predict the labels for training data
y_predict=knn.predict(X_test)

# using .score() method the accuracy of classifier's prediction can be computed and printed
print("score=",knn.score(X_test, y_test))