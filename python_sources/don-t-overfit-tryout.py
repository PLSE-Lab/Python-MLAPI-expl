#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Importing the necessary libraries
import numpy as np 
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical

import os
print(os.listdir("../input"))


# In[12]:


#Reading the data
data = pd.read_csv("../input/train.csv")


# In[13]:


#Checking the data
data.head


# In[14]:


#Checking the data
column_names = list(data.columns.values)
print("Shape: ", data.shape)
print(column_names)


# In[15]:


#Printing the target values
print(data["target"].unique())


# In[16]:


#Checking for null entries
data.isnull().any()


# In[17]:


#Preparing the input and the output for the training
X = data.drop(["id", "target"], axis = 1)
Y = data["target"]
#Y = to_categorical(Y)
print(X.shape, " ", Y.shape)


# In[18]:


#Defining a simple neural network model for the task of classification
model = keras.models.Sequential()
layer1 = keras.layers.Dense(512, input_shape = [300], activation = "relu")
model.add(layer1)
layer2 = keras.layers.Dense(256, activation = "relu")
model.add(layer2)
layer3 = keras.layers.Dense(128, activation = "relu")
model.add(layer3)
layer4 = keras.layers.Dense(56, activation = "sigmoid")
model.add(layer4)
layer5 = keras.layers.Dense(28, activation = "sigmoid")
model.add(layer5)
layer6 = keras.layers.Dense(10, activation = "sigmoid")
model.add(layer6)
#layer7 = keras.layers.Dense(2, activation = "softmax")
layer7 = keras.layers.Dense(1, activation = "sigmoid")
model.add(layer7)


# In[19]:


#Compiling the model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
print(model.summary())


# In[20]:


trained_model = model.fit(X, Y, batch_size = 50, epochs = 8000)


# In[21]:


#Saving the model
model.save("trained_model.h5")


# In[22]:


#Performing prediction on the test data
test_data = pd.read_csv("../input/test.csv")
test_data = test_data.drop(["id"], axis = 1)
#model = keras.model.load_model("trained_model.h5")
predictions = model.predict(test_data, batch_size = 50)
'''
final_pred = []
for pred in predictions:
    final_pred.append(0*pred[0]+1*pred[1])
'''


# In[23]:


#submitting the predictions
submission = pd.read_csv("../input/sample_submission.csv")
#submission["target"] = final_pred
submission["target"] = predictions

print(submission.head())


# In[24]:


submission.to_csv("submission.csv", index = False)

