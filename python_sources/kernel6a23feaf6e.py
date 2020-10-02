"""
    @script-author: Robin Wilson
    @script-description: KNN using Scikit on iris dataset
    @scrpt-date: 13.01.2020
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.model_selection as ms
import sklearn.preprocessing as pre
import sklearn.metrics as mms
import sklearn.neighbors as skne

from sklearn import datasets
iris = datasets.load_iris()
iris
target_var=iris.target
train_data=iris.data

X=train_data.tolist()
Y=target_var.tolist()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=ms.train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

classifier.fit(X_train,Y_train)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(Y_test)
print(list(Y_pred))


#import sklearn.metrics as mms
#mms.accuracy_score(Y_test,Y_pred)







