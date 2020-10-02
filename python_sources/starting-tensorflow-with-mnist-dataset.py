#!/usr/bin/env python
# coding: utf-8

# **What is Tensorflow?**
# 
# Tensorflow architecture works in three parts:
# 
# 1. * Preprocessing the data
# 1. * Build the model
# 1. * Train and estimate the model

# **
# Why Every Data Scientist Learn Tensorflow 2.x not Tensorflow 1.x** 
# 
# 
# 1. API Cleanup
# 2. Eager execution
# 3. No more globals
# 4. Functions, not sessions (session.run())
# 5. Use Keras layers and models to manage variables
# 6. It is faster
# 7. It takes less space
# 8. More consistent
# 
# and many more, watch Google I/O https://www.youtube.com/watch?v=lEljKc9ZtU8
# Github Link: https://github.com/tensorflow/tensorflow/releases

# **Importing the MNIST Dataset**
# 
# Fashion MNIST dataset contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels)

# In[ ]:


import tensorflow as tf
from tensorflow import keras


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

(x_train,y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
# In[ ]:


print(type(x_train))
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[ ]:


print(np.max(x_train))
print(np.mean(x_train))


# In[ ]:


print(y_train)


# In[ ]:


class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


# In[ ]:


plt.figure()
plt.imshow(x_train[1])
plt.colorbar()


# In[ ]:


x_train = x_train/255.0
x_test = x_test/255.0


# **Build the Model with TF 2.0**

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


# In[ ]:


model = Sequential()
model.add(Flatten(input_shape = (28,28) ))
model.add(Dense(128, activation= 'relu') )
model.add(Dense(10, activation = 'softmax'))


# In[ ]:


model.summary()


# **
# Model Compilation**
# 
# * Loss Function
# * Optimizer
# * Metrics

# In[ ]:


model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.fit(x_train, y_train, epochs = 10)


# **Further  Analysis after Training**

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


y_pred = model.predict_classes(x_test)


# In[ ]:


print(y_pred)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


pred = model.predict(x_test)


# In[ ]:


print(pred)


# In[ ]:


print(pred[0])
print(y_pred[0])
print(np.argmax(pred[0]))


# In[ ]:




