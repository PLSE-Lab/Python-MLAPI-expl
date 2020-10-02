#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 23:07:29 2018

@author: somesh
"""

from scipy.spatial.distance import euclidean as euc

class KNN:
    def fit(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
    def closest(self, row):
        best = euc(row, self.train_X[0])
        best_index = 0
        for i in range(1, len(self.train_X)):
            temp = euc(row, self.train_X[i])
            if temp<best:
                best = temp
                best_index = i
        return self.train_Y[best_index]
    def predict(self, val_X):
        pred = []
        for i in val_X:
            label = self.closest(i)
            pred.append(label)
        return pred
    
        

from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
train_X, train_y, val_X, val_y = train_test_split(X, y, test_size=0.3)
model = KNN()
model.fit(train_X, val_X)
pred = model.predict(train_y)
print(accuracy_score(pred, val_y)*100, "%")


#Plotting (Scatter Plots)
iris = sns.load_dataset("iris")
iris["ID"] = iris.index
iris["ratio"] = iris["sepal_length"]/iris["sepal_width"]
sns.lmplot(x="ID", y="ratio", data=iris, hue="species", fit_reg=False, legend=False)

plt.legend()
plt.show()