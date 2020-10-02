# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv as csv


train = pd.read_csv('../input/train.csv')
train.set_index('PassengerId',inplace = True, drop = True)
test = pd.read_csv('../input/test.csv')
test_ids = test['PassengerId']
test.set_index('PassengerId',inplace = True, drop = True)

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex


def parse_model_0(X):
    if hasattr(X, 'Survived'):
        target = X.Survived
    X['Fare'] = (X.Fare.fillna(X.Fare.median()) - np.std(X.Fare.fillna(X.Fare.median()))) \
    /(X.Fare.fillna(X.Fare.median()).mean())
    X['Family'] =  X["Parch"] + X["SibSp"]
    X['Family'].loc[X['Family'] > 0] = 1
    X['Family'].loc[X['Family'] == 0] = 0
    X['Gender'] = X['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    X['Person'] = X[['Age','Sex']].apply(get_person,axis=1)
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = X[(X['Gender'] == i) & \
            (X['Pclass'] == j+1)]['Age'].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            X.loc[ (X.Age.isnull()) & (X.Gender == i) & (X.Pclass == j+1),\
                'Age'] = median_ages[i,j]
    X["Age"][np.isnan(X["Age"])] = X.Age.median()  
    X['Age'] = (X.Age - np.std(X.Age))/(X.Age.mean())
    to_dummy = ['Pclass','Person']
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix = 'Split')
        for col in split_temp:
            X[col] = split_temp[col]
        del X[dum]
    to_del = ['Name','Ticket','Cabin','Embarked','Sex','SibSp','Parch','Gender']
    for col in to_del : del X[col]
    if hasattr(X, 'Survived'):
        del X['Survived']
        return X, target
    return X
    
def compute_score(clf, X,y):
    Xval = cross_val_score(clf, X,y, cv = 10, n_jobs = 10)
    return np.mean(Xval)
    

def clf_importance(X,clf):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
            print(("%d. feature : %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]])))

X, y = parse_model_0(train.copy())
Xtest = parse_model_0(test.copy())
rf = RandomForestClassifier(n_estimators = 100,n_jobs = -1, max_depth = 8)
rf.fit(X,y)
clf_importance(X, rf)


print(compute_score(rf,X,y))
predictions = rf.predict(Xtest).astype(int)
predictions_file = open("rf1.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(test_ids, predictions))
predictions_file.close()




