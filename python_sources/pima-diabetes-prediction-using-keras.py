#!/usr/bin/env python
# coding: utf-8

# # Set Up

# In[1]:


# imports
import pandas as pd
import numpy as np


# In[2]:


# read data from csv file
data = pd.read_csv('../input/diabetes.csv')


# In[6]:


#data description
data.describe(include="all")


# # Data Splitting

# In[4]:


X = data.iloc[:,0:8]
y = data.iloc[:,8]


# In[5]:


#Spliting our dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[6]:


#Data scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform((X_train))
X_test = scaler.fit_transform((X_test))


# # Training

# In[7]:


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[32]:


model = Sequential([
    Dense(8, input_shape=(8,), activation='relu'),
    Dense(6, activation='relu'),
    Dense(2, activation='softmax')
])


# In[33]:


model.summary()


# In[54]:


model.compile(Adam(lr=.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[55]:


model.fit(X_train, y_train, batch_size=10, epochs=20, shuffle=True, verbose=2)


# # Prediction

# In[56]:


y_pred = model.predict_classes(X_test, batch_size=10, verbose=0)


# In[57]:


#Accuracy score
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred)


# In[58]:


print("accuracy: ", acc)


# In[ ]:




