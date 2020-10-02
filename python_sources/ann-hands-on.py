#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('../input/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_encoder_Country = LabelEncoder()
X[:, 1] =lbl_encoder_Country.fit_transform(X[:, 1])
lbl_encoder_gender = LabelEncoder()
X[:, 2] = lbl_encoder_gender.fit_transform(X[:, 2])
onehotencoder_country = OneHotEncoder(categorical_features=[1])
X = onehotencoder_country.fit_transform(X).toarray()
# dropping Ist dummy variable of country to avoid dummy variable trap
X = X[:, 1:]


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(copy=True, with_mean=True, with_std=True) # calcuating the Z score
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[ ]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


# Initializing the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim=11))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size= 10 , epochs=100 )



# In[ ]:


# Predicting the Test set results
y_pred =  classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(y_pred)


# In[ ]:


# Making the Confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 619, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction >= 0.5)
print(new_prediction)

