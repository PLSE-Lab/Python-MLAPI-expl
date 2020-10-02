#!/usr/bin/env python
# coding: utf-8

#  **This is an interesting dataset for building Deep Learning Neural Networks. here we use tensorflow keras API to form the model.**

# In[ ]:


# Import the necessary libraries

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
import os
import random


# In[ ]:


# Set the path of the input folder 

data = "../input/flowers/flowers/"

# List out the directories inside the main input folder

folders = os.listdir(data)

print(folders)


# In[ ]:


# Import the images and resize them to a 128*128 size
# Also generate the corresponding labels

image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue
      


# In[ ]:


# Transform the image array to a numpy type

train = np.array(train_images)

train.shape


# In[ ]:


# Reduce the RGB values between 0 and 1

train = train.astype('float32') / 255.0


# In[ ]:


# Extract the labels

label_dummies = pandas.get_dummies(train_labels)

labels =  label_dummies.values.argmax(1)


# In[ ]:


pandas.unique(train_labels)


# In[ ]:


pandas.unique(labels)


# In[ ]:


# Shuffle the labels and images randomly for better results

union_list = list(zip(train, labels))
random.shuffle(union_list)
train,labels = zip(*union_list)

# Convert the shuffled list to numpy array type

train = np.array(train)
labels = np.array(labels)


# In[ ]:


# Develop a sequential model using tensorflow keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Dense(128, activation=tf.nn.tanh),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])


# In[ ]:


# Compute the model parameters

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Train the model  with 5 epochs 

model.fit(train,labels, epochs=5)

