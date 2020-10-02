# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:24:25 2019

@author: Shari
""" 

#   Tic -Tac - Toe problem using Logistic regression 

#Import librarues
import numpy as np
import pandas as pd
data=pd.read_csv("../input/tic-tac-toe.csv")
data.head(5)
data_new=pd.get_dummies(data.ix[:,0:9])
data_new.head()
data_final=pd.concat([data_new,data.ix[:,9]],axis=1)

#Splitting data into train & test
from sklearn.model_selection import train_test_split 
train,test=train_test_split(data_final,test_size=0.3)

train.head()

#Applying logistic regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()

x_train=train.ix[:,:-1]
y_train=train.ix[:,27]
x_test=test.ix[:,:-1]
y_test=test.ix[:,27]


LR=LR.fit(x_train,y_train)

LR.score(x_train,y_train)

#Evaluate model on test data

from sklearn.model_selection import cross_val_score
from sklearn import metrics

pred=LR.predict_proba(x_test)
predicted=LR.predict(x_test)

#Finding accuracy 
print("Accuracy ",metrics.accuracy_score(y_test,predicted))

print("Classification report",metrics.classification_report(y_test,predicted))

print("Confusion matrix",metrics.confusion_matrix(y_test,predicted))

#Finding accuracy with cross validation
scores=cross_val_score(LR,x_train,y_train,cv=10,scoring='accuracy')
print(scores)
print(scores.mean())