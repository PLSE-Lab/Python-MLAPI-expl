#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import tensorflow as tf
from tensorflow import keras
#Version of tensorflow
print(tf.__version__)


# In[ ]:


file_name = "/kaggle/input/digit-recognizer/train.csv"
data = pd.read_csv(file_name)
target = pd.DataFrame(data['label'])
features = data.drop('label',axis=1)
print(data.shape)
print(data.describe)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import tensorflow as tf
from tensorflow import keras
#Building a neural network
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(128,input_dim=784,activation=tf.nn.relu))
#model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(21,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


# In[ ]:


model_23 = tf.keras.models.Sequential([
                            tf.keras.layers.Dense(132,input_dim=784,activation = tf.nn.relu),
                            tf.keras.layers.Dense(40,activation=tf.nn.relu),
                            tf.keras.layers.Dense(10,activation=tf.nn.sigmoid)
    
])


# In[ ]:





# In[ ]:


model_23.summary()


# In[ ]:


#Compliing the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[ ]:





#  taking another model but using the Stochastic greadient descent algorithm as an optimizer

# In[ ]:


model_23.compile(optimizer="sgd",loss="sparse_categorical_crossentropy",metrics=['accuracy'])


# In[ ]:


#Fit the model i.e find the patterns in the training data
model.fit(features,target,epochs=8)


# In[ ]:


model_23.fit(features,target,epochs=10)


# In[ ]:


scores = model.evaluate(features,target)
print(scores)


# In[ ]:


scores_23 = model_23.evaluate(features,target)


# In[ ]:


#testing data
test_data=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test_target = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
s = model.evaluate(test_data,test_target['Label'])


# In[ ]:


features , test_data = np.array(features),np.array(test_data)
features,test_data = features.reshape(42000,28,28,1),test_data.reshape(28000,28,28,1)


# In[ ]:


#Building a convolutional layer 
model_update = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
                                           tf.keras.layers.MaxPooling2D(2,2),
                                           tf.keras.layers.Flatten(),
                                           tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                           tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                                            ])


# In[ ]:


model_update.summary()
# Explains how the convolutions are acting on the images by concentrating on the main features
# and pooling helps in compressing the image but keeping the features intact


# In[ ]:


# Compiling the Convolutional model
model_update.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=['accuracy'])


# In[ ]:


model_update.fit(features,target,epochs=5)


# In[ ]:





# In[ ]:


# Evaluating the test data with a convolutional model, I hope there is an improvement
model_update.evaluate(test_data,np.array(test_target['Label']))


# In[ ]:


# Not much of an improvement 


# Lets try normalizing the features and test_data which contain the pixel values 
# of the pictures being feeded into the Convolutional Neural Network
# 

# In[ ]:


#Normalizing
features , test_data = features , test_data
#Converting features and test_data into numpy arrays
features , test_data = np.array(features),np.array(test_data)
#Reshaping them
features , test_data = features.reshape(42000,28,28,1),test_data.reshape(28000,28,28,1)
"""
features , test_data = np.array(features),np.array(test_data)
features,test_data = features.reshape(42000,28,28,1),test_data.reshape(28000,28,28,1)
"""


# In[ ]:


#Building a Convolutional Neural Network with 3 Convolutional layers with their respective pooling
model_3 = tf.keras.models.Sequential([
                                tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(1028,activation=tf.nn.relu),
                                tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                                     ])


# In[ ]:


#Compiling the model
model_3.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


# Fit the model to find the patterns
model_3.fit(features,target,epochs=10)


# Evaluating our model on the training data (which is actually not a good idea as our model has learnt the features to look out from the training data)

# In[ ]:


model_3.evaluate(features,target)


# In[ ]:


model_3.evaluate(test_data,np.array(test_target['Label']))


# Lets try another Convolutional Neural Network having 3 Convolutional layers along with Max Pooling layers

# In[ ]:


model_4 = tf.keras.models.Sequential([
                                tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
                                tf.keras.layers.MaxPooling2D(2,2),
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512,activation = tf.nn.relu),
                                tf.keras.layers.Dense(10,activation = tf.nn.softmax)
                                ])


# In[ ]:


# Helping us how the Convolutional layers and MaxPooling layers reduce the parameters by concentrating
#on the main features and copressing the image by keeping the features  intact
model_4.summary()


# In[ ]:


#model compiling bysetting up the optimizer(which tries to minimize the error by making sure that the weights and baises 
#helps in reaching the global minima or  good enough local minima)

model_4.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


# Find the main features by update the weights and biases of the network
model_4.fit(features,target,batch_size=32,epochs=5)


# In[ ]:


model_4.evaluate(test_data,test_target['Label'])


# In[ ]:




