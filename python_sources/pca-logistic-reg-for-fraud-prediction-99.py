### Credit Card Fraud Prediction Analysis

## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing the dataset
dataset = pd.read_csv('creditcard.csv')
X = dataset.iloc[:,1:29]
Y = dataset.iloc[:,-1]
X_Time_Amount = dataset.iloc[:,[0,29]]

# Feature Selection 1
from sklearn.decomposition import PCA
PCA = PCA(n_components = 20)
X = pd.DataFrame(PCA.fit_transform(X))
explained_variance = PCA.explained_variance_ratio_

# Time and Amount Values
Time = pd.DataFrame(np.zeros((len(Y),1)))
Aux = X_Time_Amount.iloc[1:,0].reset_index(drop=True)
Time.iloc[1:,0] = Aux - X_Time_Amount.iloc[:-1,0]
Time.columns = ['Time']

X = pd.concat([X, Time, X_Time_Amount.iloc[:,1]], axis = 1)

## Finding Class = 1
indexes = []
for i in range(0,len(Y)):
   if Y[i]==1:
       indexes.append(i)

## New dataset
# Adding 1's
X_new = X.iloc[indexes].reset_index(drop=True)
Y_new = Y.iloc[indexes].reset_index(drop=True)

# Adding 0's
import random
k = 15000 # amount of samples
rdm = random.sample(range(1,len(Y)), k) # generating random indexes
X_new = pd.concat([X_new, X.iloc[rdm,:]]).reset_index(drop=True) # establishing feature matrixes
Y_new = pd.concat([Y_new, Y[rdm]]).reset_index(drop=True)

# Removing Repeated Rows
indexes = []
for i in range(492,len(Y_new)): # finds 1's after index 491
    if Y_new.iloc[i]==1:
        indexes.append(i)
X_new = X_new.drop(indexes) # drops repeated rows
Y_new = Y_new.drop(indexes)

## Splitting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size = 0.25)

# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression(random_state = 0)
Classifier.fit(X_train, Y_train)

# Predicting the test set results
from sklearn.metrics import accuracy_score
Y_pred = Classifier.predict(X_test)
Accuracy = accuracy_score(Y_test, Y_pred)