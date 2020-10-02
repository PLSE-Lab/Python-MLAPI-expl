# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv as csv

train = pd.read_csv('../input/train.csv')
train.set_index('PassengerId',inplace = True, drop = True)
test = pd.read_csv('../input/test.csv')
test_ids = test['PassengerId']
test.set_index('PassengerId',inplace = True, drop = True)

def parse_model_0(X):
    if hasattr(X, 'Survived'):
        target = X.Survived
    to_dummy = ['Pclass','Sex']
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix = 'Split')
        for col in split_temp:
            X[col] = split_temp[col]
        del X[dum]
    X['Age'] = X.Age.fillna(X.Age.median())
    X['is_child'] = X.Age < 8
    X['Fare'] = X.Fare.fillna(X.Fare.median())
    to_del = ['Name','Cabin','Embarked','Ticket']
    for col in to_del : del X[col]
    if hasattr(X, 'Survived'):
        del X['Survived']
        return X, target
    return X
    
X, y = parse_model_0(train.copy())
Xtest = parse_model_0(test.copy())
lr = LogisticRegression()
lr.fit(X,y)
predictions = lr.predict(Xtest).astype(int)

predictions_file = open("linearReg1.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_ids, predictions))
predictions_file.close()




