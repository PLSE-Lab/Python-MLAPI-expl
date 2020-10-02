# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/heart.csv')
df = df.fillna(0)
df.columns
y = df['target'].values
df = df.fillna(0)
X = df.drop(columns=['target'], axis=1).values

#Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = GradientBoostingClassifier(random_state=39, n_estimators=50)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')

#Model with RandomForest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)
model = RandomForestClassifier(random_state=39, n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')

#With more estimators Random Forest has the same accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
model = RandomForestClassifier(random_state=33, n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = np.mean(pred == y_test)
print('accuracy: ', accuracy*100, '%')