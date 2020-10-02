# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 10:49:14 2017

@author: Netik
"""

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load dataset

url = "../input/creditcard.csv"
colnames = ("Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12"
            ,"V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24",
            "V25","V26","V27","V28","Amount","class")
dataset = pd.read_csv(url, names = colnames, index_col = 0)
idata = dataset.reset_index()
idata.drop('Time', axis = 1, inplace = True)
idata.drop('Amount', axis = 1, inplace = True)
idata.drop('class', axis = 1, inplace = True)
#removing row with cloumn label
idata.drop(idata.index[0], inplace = True)
idata = idata.astype(np.float)

odata = dataset.reset_index()
odata = odata['class']
odata.drop(odata.index[0], inplace = True)
odata = odata.astype(np.float)

idata, i_test, odata, o_test = model_selection.train_test_split(idata, odata, test_size=0.20)
print(idata.describe())

X = idata.values
Y = odata.values
kfold = model_selection.KFold(n_splits=5)
knn = LinearDiscriminantAnalysis()# KNeighborsClassifier(n_neighbors = 35)
knn.fit(X,Y)

results = model_selection.cross_val_score(knn, X, Y, cv=kfold)
print("without scaling:",results.mean()*100.0)
predictions = knn.predict(i_test)
print("accuracy on test:",accuracy_score(o_test, predictions)*100)

#idata = preprocessing.MinMaxScaler().fit_transform(idata)
#X = idata
#results = model_selection.cross_val_score(knn, X, Y, cv=kfold)
##predictions = knn.predict(i_test)
#print("with scaling:", accuracy_score(o_test, predictions)*100.0)


