#!/usr/bin/env python
# coding: utf-8

# **Load your data**

# In[ ]:


import pandas 
from keras.models import Sequential
from keras.layers import Dense

import numpy
numpy.random.seed(7)

data= pandas.read_csv('../input/Internship Application  - Classification.csv')

data.head()


# Using **Slicing** to store values into input and output.

# In[ ]:


data = data.values
X = data[:,0:5]
Y = data[:,5]


# From **scikit learn**, we use **train_test_split** to allocate values to our input and output. It is best to also randomise your data.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1
                                                    ,random_state = 0)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.11
                                                   ,random_state = 0)


# We now see the shapes of all the inputs; the **train set** , the **validation set** and the **test set**

# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_val.shape


# **Create the model using keras**
# 

# In[ ]:


model = Sequential()

model.add(Dense(10, input_dim=7, kernel_initializer='uniform', activation='relu'))

model.add(Dense(4, kernel_initializer='uniform', activation='relu'))

model.add(Dense(3, kernel_initializer='uniform', activation='relu'))

model.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))

model.summary()


# **Compiling the model**
# 
# **Binary Cross entropy** is used because the outcome of the data set is binary classification

# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

