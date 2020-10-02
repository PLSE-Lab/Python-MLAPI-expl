# -*- coding: utf-8 -*-
"""
@author: Somil Shah
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Scanning the dataset to be used and selecting the columns needed as input and output
dataset=pd.read_csv("../input/data.csv")
X=dataset.iloc[:,2:32].values
Y=dataset.iloc[:,[1,]].values

#Label encoding the categorical outputs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y_1 = LabelEncoder()
Y[:,0] = labelencoder_Y_1.fit_transform(Y[:,0])
Y=Y.astype(float)

#Scaling all the inputs to values between -1 and 1 to avoid biasing of input values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)

#Splitting data into training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True)

#Creating the neural network with 30 as input units and 2 hidden layers with 16 units
classifier = Sequential()

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

classifier.add(Dropout(0.4))

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dropout(0.2))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, batch_size = 25, epochs = 50,validation_data=(X_test,Y_test))

#Predicting the data on untrained set
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
