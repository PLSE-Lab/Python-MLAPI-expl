#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.datasets import boston_housing


# 
# - Loading the dataset
# - Splitting the datasets into training and test sets

# In[ ]:


(X_train,y_train),(X_test,y_test) = boston_housing.load_data()


# # Building the model

# In[ ]:


model = Sequential()


# In[ ]:


model.add(Dense(64,activation='relu',input_shape=(13,)))
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))


# # Compiling the model

# In[ ]:


model.compile(loss='mean_squared_error',optimizer='RMSProp',metrics=['mse'])


# - Summary of the model

# In[ ]:


model.summary()


# - Training the model

# In[ ]:


history = model.fit(X_train,y_train,epochs=100,validation_split=0.10)


# In[ ]:


x= np.sqrt(np.mean(history.history['val_mse']))
print(100-x)


# In[ ]:




