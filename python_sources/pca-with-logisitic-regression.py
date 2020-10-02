# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:13:51 2016

@author: Home
"""

import pandas as pd
from matplotlib import pyplot as py
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
# The competition datafiles are in the directory ../input
# Read competition data files:
# train = pd.read_csv("train.csv")
# test  = pd.read_csv("test.csv")

def train_models(data,target):
    models=[]
    y_train=pd.get_dummies(target)
    for i in range(6,10):
         clf=LogisticRegression()
         model=clf.fit(data,y_train.loc[:,i])
         joblib.dump(clf, 'filename'+str(i+1)+'.pkl',protocol=2) 
         models.append(model)
    return models
# Write to the log:
#Any files you write to the current directory get shown as outputs
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

X=train.iloc[:,1:len(train)]
y=train['label']
#print(y)
pca=PCA(n_components=50)
pca.fit(train)
data=pca.transform(train)
trained_models=train_models(X,y)
print("Models have been trained")
