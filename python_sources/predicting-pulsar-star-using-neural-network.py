#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Getting and cleaning dataset

dataset = pd.read_csv('../input/pulsar_stars.csv')
dataset.dropna()
dataset.head()


# In[3]:


# The total no of columns

dataset.columns


# In[4]:


# Getting the X_data

X_data = dataset.iloc[:,0:-1].values
X_data[0:5]


# In[5]:


# Getting the Y_data

Y_data = dataset.iloc[:,-1].values
Y_data[0:5]


# In[6]:


# Scaling the X_data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[7]:


X_data = scaler.fit_transform(X_data)
X_data[0:5]


# In[8]:


#Splitting data into training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_data,Y_data)


# In[9]:


# Importing libraries for ANN creation

from keras.models import Sequential
from keras.layers import Dense


# In[10]:


#Building ANN

classifier = Sequential()
classifier.add(Dense(10 , input_dim = 8   , activation='relu'))
classifier.add(Dense(6  , input_dim = 8 , activation = 'relu'))
classifier.add(Dense(4  , input_dim = 6 , activation = 'relu'))
classifier.add(Dense(1  , input_dim = 4 , activation = 'relu'))
classifier.add(Dense(1  , activation = 'sigmoid'))


# In[11]:


# Defining the loss and optimizer

classifier.compile(loss = 'binary_crossentropy' , optimizer='adam' , metrics = ['accuracy'])


# In[12]:


# Fitting the data to model

classifier.fit(x_train,y_train,epochs = 200 , batch_size=100 )


# **So here we get the accuracy score of 90% **

# In[21]:


# Data for prediction

X_pred = np.array([dataset.iloc[1,:-1]])
print(X_pred)


# In[22]:


# Prediciting

prediction = classifier.predict(X_pred)
print(prediction)


# In[18]:


# Getting accuracy score on testing data

classifier.evaluate(x_test,y_test)


# this one is loss and , this is accuracy

# In[ ]:




