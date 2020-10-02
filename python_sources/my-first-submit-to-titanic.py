#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:29:27 2017

@author: emajluf
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# This sub is just for encoding some categorical variables
def Prepara (df):
    df['GroupSize'] = df['SibSp'] + df['Parch'] 
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    
    df = pd.get_dummies(df, columns=['Sex', 'Cabin', 'Embarked'])
    df.drop(['SibSp', 'Parch', 'Sex_male',  'Embarked_S'], axis=1, inplace=True)
    return df

# Load files and purge no-way variables
dfTrain = pd.read_csv('../input/train.csv')
dfTest = pd.read_csv('../input/test.csv')

dfTrain.drop(['PassengerId', 'Name', 'Ticket' ], axis=1, inplace=True)
dfTest.drop(['Name', 'Ticket' ], axis=1, inplace=True)


# Encode stuff
dfTra = Prepara(dfTrain)
dfTst = Prepara(dfTest)

X = dfTra.drop(['Survived'], axis=1)
y = dfTra['Survived']


# Setup estimators and others
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
lasso = Lasso(alpha=0.1, normalize=True)
knn = KNeighborsClassifier(n_neighbors=6)
xgb = XGBClassifier()
pipeline = make_pipeline(imp, xgb)

# Split , fit and predict test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=68)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))


# Evaluar resultados
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Alinear resultados
result = pd.DataFrame(y_test)
result['pred'] = y_pred


"""
Now, process test set
"""

submission = pd.DataFrame( {
    "PassengerId": dfTst["PassengerId"],            
    "Survived": pipeline.predict(dfTst)
            } ) 
submission.to_csv('TitanicSubmit.csv', index=False)   
