# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:23:05 2019

@author: Arun
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

# Importing the dataset
traindataset = pd.read_csv('../input/training.csv')
testdataset = pd.read_csv('../input/testing.csv')
X_train = traindataset.iloc[:, 1:].values
y_train = traindataset.iloc[:, 0].values

X_test = testdataset.iloc[:, 1:].values
y_test = testdataset.iloc[:, 0].values

print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling using robust scaler in case of outliers
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
#using lasso regularizer to select important features
classifier = LogisticRegression(random_state = 0, penalty='l1', C= 4.0)
classifier.fit(X_train, y_train)
print(y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#curves
train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

train_sizes, train_scores, validation_scores = learning_curve(
                                                   estimator = LogisticRegression(random_state = 0, penalty='l1'), X = X_train,
                                                   y = y_train, train_sizes = train_sizes, cv = 5, scoring = 'accuracy')

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)

print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
print('\n', '-' * 20) # separator
print('\nMean validation scores\n\n',pd.Series(validation_scores_mean, index = train_sizes))
import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.plot(train_sizes, train_scores_mean, label = 'Training Accuracy')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation Accuracy')

plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,2)