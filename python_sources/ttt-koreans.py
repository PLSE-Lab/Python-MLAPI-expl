# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
df = pd.read_csv('../input/tttkoreans/train.csv', header=0) # load data from csv
df_test = pd.read_csv('../input/ttttest/test.csv', header=0) # load data from csv
cols = ['Name','Ticket','Cabin']
cols_test = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)
df_test = df_test.drop(cols_test,axis=1)
df = df.dropna()

dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
      dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
df = pd.concat((df,titanic_dummies),axis=1)
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
df['Age'] = df['Age'].interpolate()

dummies_test = []
cols_test = ['Pclass','Sex','Embarked']
for col_test in cols_test: 
        dummies_test.append(pd.get_dummies(df_test[col_test]))
titanic_dummies_test = pd.concat(dummies_test, axis=1)
df_test = pd.concat((df_test,titanic_dummies_test),axis=1)
df_test = df_test.drop(['Pclass','Sex','Embarked'],axis=1)
df_test['Age'] = df_test['Age'].interpolate()


#Train and Test dataset 
X_train = df[df.columns.difference(['Survived'])].values
y_train = df['Survived'].values
X_test = df_test.values
df.info()
df_test.info()

#Random forest
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit (X_train, y_train)


#Prediction
y_pred = clf.predict(X_test)
y_pred = np.round(y_pred).astype(int)
y_pred = y_pred.ravel()

