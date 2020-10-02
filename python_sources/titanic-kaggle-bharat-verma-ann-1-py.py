# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:07:30 2017

@author: Bharat
"""

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

#Lets lets the ANN
#--part2
#lets import keres library
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN (input layer)
classifier = Sequential()

#adding input layer and first hidden layer
#how to choose the no. of nodes in the hidden layers: as the average 
#of no of nodes in the input layer and the no of nodes in the output layer 
#no of nodes in input layer for us is 11(No of independent var)
#no of nodes in output layer for us is 1(No of dependent var)
#for the first hidden layer we need to tell the model in input_dim parameter
#that how many nodes are there
#activation function for hidden layer is rectifier
#activation function for output layer is sigmoid
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 4))

#adding the second hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

#adding the output layer
#if u have more than 3 cases in dependent variables then use softmax activation function
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling the ANN
#binary_crossentropy -> loss func (dependent var has binary output)
#categorical_crossentropy -> loss func (dependent var has more tha 2 output)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', matrics= ['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_supply, Y, batch_size = 10, nb_epoch = 200)

#part 3- making the predictions and evaluating the model

#predict the weather the passanger will survive or die
Y_pred = classifier.predict(X_test_supply)
Y_pred = (Y_pred > 0.4)

#~Bharat Titanic evaluation ANN
