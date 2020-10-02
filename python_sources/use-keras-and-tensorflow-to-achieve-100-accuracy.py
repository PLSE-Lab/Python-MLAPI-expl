#!/usr/bin/env python
# coding: utf-8

# Load libraries

# In[ ]:


from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Load iris dataset

# In[ ]:


iris = datasets.load_iris()


# Split dataset into train dataset and test dataset

# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)


# Build a neural network model

# In[ ]:


model = tf.keras.models.Sequential([tf.keras.layers.Dense(4, activation=tf.nn.relu), tf.keras.layers.Dense(3, activation=tf.nn.softmax)])


# Compile the model

# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Start to train the model.

# In[ ]:


model.fit(X_train, y_train, epochs=1000)


# Make prediction.

# In[ ]:


pred = model.predict(X_test).argmax(axis=1)
pred


# Evalute the prediction result.

# In[ ]:


(pred == y_test).sum()/len(pred)


# **Wonderful! Achived 100% accuracy!**

# 
