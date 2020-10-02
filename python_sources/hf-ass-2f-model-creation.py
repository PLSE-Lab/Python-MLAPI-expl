#!/usr/bin/env python
# coding: utf-8

# **Load the data**

# In[15]:


import pandas 
from keras.models import Sequential
from keras.layers import Dense

import numpy
numpy.random.seed(7)

data= pandas.read_csv('../input/Road Accidents-Regression.csv')

data.head()


# **Slicing** the dataset to get input and output values

# In[16]:


data = data.values
X = data[:,0:5]
Y = data[:,5]


# From **scikit learn**, we use **train_test_split** to allocate values to our input and output. It is best to also randomise your data.

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1
                                                    ,random_state = 0)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11
                                                   ,random_state = 0)


# We now see the shapes of all the inputs; the **train set** , the **validation set** and the **test set**

# In[18]:


X_train.shape


# In[19]:


X_val.shape


# In[20]:


X_test.shape


# **Create the model**

# In[21]:


model = Sequential()

model.add(Dense(11, input_dim=7, kernel_initializer='uniform', activation='relu'))

model.add(Dense(5, kernel_initializer='uniform', activation='relu'))

model.add(Dense(3, kernel_initializer='uniform', activation='relu'))

model.add(Dense(2, kernel_initializer='uniform'))

model.summary()


# **Compiling the model**
# 
# Using mean squared error as a loss function because our outcome is Regression

# In[22]:


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

