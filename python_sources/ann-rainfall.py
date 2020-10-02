# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")
os.getcwd()

#Importing Dataset
dataset = pd.read_csv("../input/weatherAUS.csv")

#Preprocessing
    #Viewing all the columns information
dataset.info()          
    #Sorting columns by their count in ascending order
dataset.count().sort_values()       #
    #Dropping columns with less records column wise
dataset = dataset.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am','RISK_MM'], axis = 1)
    #Deleting records with null values
dataset = dataset.dropna(axis = 'rows')

#X & y
X = dataset.iloc[:, 2:18]
y = dataset.iloc[:, 18]
data = dataset.head(1000)


            ###LabelEncoding###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""WindGustDir"""
le = LabelEncoder()
X['WindGustDir'] = le.fit_transform(X['WindGustDir'])

"""WindDir9am"""
le1 = LabelEncoder()
X['WindDir9am'] = le1.fit_transform(X['WindDir9am'])

"""WindDir3pm"""
le2 = LabelEncoder()
X['WindDir3pm'] = le2.fit_transform(X['WindDir3pm'])

"""RainToday"""
le3 = LabelEncoder()
X['RainToday'] = le3.fit_transform(X['RainToday'])

            ###OneHotEncoder###
"""WindGustDir"""
ohe = OneHotEncoder(categorical_features=[3])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

"""WindDir9am"""
ohe1 = OneHotEncoder(categorical_features=[19])
X = ohe1.fit_transform(X).toarray()
X = X[:,1:]

"""WindDir3pm"""
ohe2 = OneHotEncoder(categorical_features=[33])
X = ohe2.fit_transform(X).toarray()
X = X[:,1:]

"""RainToday"""
ohe3 = OneHotEncoder(categorical_features=[55])
X = ohe3.fit_transform(X).toarray()
X = X[:,1:]


#splitting the data into training & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Standardization
from sklearn.preprocessing import StandardScaler
ss  = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
y_train = (y_train == 'Yes')

        ###ANN Regression###
#Fitting the classifier to the dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#Adding 2 HL
classifier.add(Dense(input_dim = 108, activation = 'relu', kernel_initializer = 'uniform', output_dim = 55))
classifier.add(Dense(activation = 'relu', kernel_initializer = 'uniform', output_dim = 55))
classifier.add(Dense(activation = 'sigmoid', kernel_initializer = 'uniform', output_dim = 1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, epochs = 32, batch_size = 10)

#Predicting the target variable
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_test = (y_test == 'Yes')

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

