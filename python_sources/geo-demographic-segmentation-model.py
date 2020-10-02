# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 23:31:55 2018

@author: Deathrow77
"""

import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('../input/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Handling Missing Values

# Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_1 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(X[:, 1])
label_encoder_2 = LabelEncoder()
X[:, 2] = label_encoder_2.fit_transform(X[:, 2])
# Moderating the Modal Difference between the categorical Features
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X =one_hot_encoder.fit_transform(X).toarray()
# Deleting the extra dummy variable
X = X[:, 1:]

#Performing the train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)


# Importing Keras and Building the ANN
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann = Sequential()

# Adding Layers to the ANN

ann.add(Dense(6, input_shape=(11,), kernel_initializer='uniform', activation='relu'))
ann.add(Dense(6, kernel_initializer='uniform', activation='relu'))
# Using Sigmoid Activation Function to get a Probabilistic Output
ann.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#Compile the ANN
ann.compile(optimizer='adam', metrics=['accuracy'], loss=['binary_crossentropy'])
# Training the ANN
ann.fit(X_train, y_train,batch_size=8, epochs=10)
# Predicting the values
y_pred = ann.predict(X_test)
# Creating Categorical data by Selecting the values with Probability > 0.5 
y_pred = (y_pred>0.5)

# Creating Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)


