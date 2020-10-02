#!/usr/bin/env python
# coding: utf-8

# In[1]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # Shallow Net in Keras(High level API to call Tensorflow)[](http://)

# **Load dependancy**

# In[2]:


import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# **Loading dataset**

# In[3]:


data_train = pd.read_csv('../input/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashion-mnist_test.csv')
data_train.shape, data_test.shape


# **Spliting and preprecessing the training and test dataset**

# In[4]:


X_train = np.array(data_train.iloc[:, 1:]).astype('float32')
y_train = np.array(data_train.iloc[:, 0])
X_test = np.array(data_test.iloc[:, 1:]).astype('float32')
y_test = np.array(data_test.iloc[:, 0])
X_train = X_train / 255
X_test = X_test / 255


# Determining the categories in dataset

# In[5]:


np.unique(y_train)


# **Implementing one hot encoded using utilities in Keras**

# In[6]:


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)


# ** Building Sequential neural network architecture with only one hidden layer and  **

# In[7]:


model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))


# In[8]:


model.summary()
print((64*784)+64)
print((10*64) + 10)


# **Configure model**

# In[9]:


model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])


# Training the model

# In[11]:


model.fit(X_train, y_train, batch_size=128, verbose = 0, epochs=40, validation_data=(X_test, y_test))


# **So, after 40 runs on the dataset, the model has Training accuracy as 0.6482 and Test Accuracy as 0.6455. **

# # Some basics and applying them in the notebook
# ## Neaurons
# * Perceptron
# * Sigmoid
# * tanh
# * ReLU
# 
# ## Cost Function
# * quadratic
# * cross-entropy
# 
# ### Gradient Descent
# ### Back Propagation
# 
# ## Layers
# * dense
# * softmax
# 

# Changin the activation to tanh because tanh and relu are always better than sigmoid. 

# In[ ]:


model = Sequential()
model.add(Dense(64, activation='tanh', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))


# After 40 runs on the dataset, the model has Training accuracy as 0.7819 and Test Accuracy as 0.7844.
# Similarly trying with relu.

# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))


# **Now reached to 0.7935 as training accuracy and 0.7927 as test accuracy. **<P>
# **lets change the cost function**

# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))


# Now reached to 0.8723 as training accuracy and 0.8655 as test accuracy.

# In[ ]:




