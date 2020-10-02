#Created on Mon Jan  26 15:18:15 2017

#@author: Bharat

#importing the libraries
#for mathematical calculations


import numpy as np
#for plotting nice charts
import matplotlib.pyplot as plt
#for importing and managing dataset
import pandas as pd

#importing the dataset
dataset = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')

#matrix of features
#creating the independent 
#and dependent variables

X = dataset.iloc[:, [2,5,7,9]].values
Y = dataset.iloc[:, 1].values
X_test = dataset_test.iloc[:, [1,4,6,8]].values



#helps to preprocessing libraries handle the missing data Imputer is the class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)


#fit this imputer object to our matrix X
imputer.fit(X[:, 0:4]) #upper bound is excluded thats why 3

#replacing the missing data by mean
X[:, 0:4] = imputer.transform(X[:, 0:4])

imputer_test = Imputer(missing_values="NaN", strategy="mean", axis=0)


#fit this imputer object to our matrix X
imputer_test.fit(X_test[:, 0:4]) #upper bound is excluded thats why 3

#replacing the missing data by mean
X_test[:, 0:4] = imputer_test.transform(X_test[:, 0:4])


# Encoding the categorical data (that are strings)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblEncoder_X = LabelEncoder()
X[:, 1] = lblEncoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray() 

lblEncoder_X_test = LabelEncoder()
X_test[:, 1] = lblEncoder_X_test.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features=[1])
X_test = onehotencoder.fit_transform(X_test).toarray() 

X_supply = X[:, 1:5]
X_test_supply = X_test[:, 1:5]

#from sklearn.feature_selection import chi2
#scores, pvalues = chi2(X_supply, Y)

#Feature scaling: so that one value doesnt dominate the other value, 
#we have to scale all the variables into same range of values.
#Future scaling for dummy variables depends on the context, and 
#we do not need to scale depedendent variable Y right now. 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_supply = sc_X.fit_transform(X_supply)


sc_X_test = StandardScaler()
X_test_supply = sc_X_test.fit_transform(X_test_supply)

#Fitting the data into to the  regression
#create your classifier
#fit the data to the classifier
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 500,
#                                    criterion = 'gini')
#classifier.fit(X, Y)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', decision_function_shape = 'ovr')
classifier.fit(X_supply, Y)

#predict the weather the passenger will survive or die
Y_pred = classifier.predict(X_test_supply)

#~Bharat