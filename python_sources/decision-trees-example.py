# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
train = pd.read_csv('/kaggle/input/ska-data-challenge-test/train.csv')
# Keep all columns except the Outcome and the Id column
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']
X = train[feature_names]
y = train[['Outcome']]
test = pd.read_csv('/kaggle/input/ska-data-challenge-test/test.csv')
testX = test[feature_names]
# If you wanted to split the training set into training+test for cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Apply scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
testX   = scaler.transform(testX)
X_scaled = np.vstack((X_train, X_test))
y_scaled = np.vstack((y_train, y_test))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_scaled, y_scaled)
# Shows how decision trees tend to overfit
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
labels = clf.predict(testX)
output = pd.DataFrame()
output['Outcome'] = labels
output['Id'] = test['Id']
output = output[['Id', 'Outcome']]
pd.DataFrame(output).to_csv("/kaggle/working/james-submission.csv", index=False)