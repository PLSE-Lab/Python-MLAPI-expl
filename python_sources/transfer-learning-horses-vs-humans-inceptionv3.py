#!/usr/bin/env python
# coding: utf-8

# Hey reader , I will be using the dataset "Horses Vs Humans" for this computer problem . 
# 
# *About the Data :
# The set contains 500 rendered images of various species of horse in various poses in various locations. It also contains 527 rendered images of humans in various poses and locations. Emphasis has been taken to ensure diversity of humans, and to that end there are both men and women as well as Asian, Black, South Asian and Caucasians present in the training set. The validation set adds 6 different figures of different gender, race and pose to ensure breadth of dat*
# 

# In[ ]:


# Importing all the required libraries

import numpy as np 
import pandas as pd 
import os
import random
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Checking the directory 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Visually Inspect Image Dataset 

input_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human'

fig, ax = plt.subplots(2, 2, figsize=(15, 7))
ax = ax.ravel()
plt.tight_layout()

for i, _set in enumerate(['train', 'validation']):
    set_path = input_path+'/'+_set
    ax[i].imshow(plt.imread(set_path+'/horses/'+os.listdir(set_path+'/horses')[0]), cmap='gray')
    ax[i].set_title('Set: {}, type:horses'.format(_set))
    ax[i+2].imshow(plt.imread(set_path+'/humans/'+os.listdir(set_path+'/humans')[0]), cmap='gray')
    ax[i+2].set_title('Set: {}, type:humans'.format(_set))
    


# In[ ]:


# Image Preprocessing

input_path = '/kaggle/input/horses-or-humans-dataset/horse-or-human/horse-or-human'

def process_data(img_dims, batch_size):
  
   
    # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    # This is fed to the network in the specified batch sizes and image dimensions
    train_gen = train_datagen.flow_from_directory(
    directory=input_path + '/train/', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)

    test_gen = test_val_datagen.flow_from_directory(
    directory=input_path + '/validation/', 
    target_size=(img_dims, img_dims), 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=True)
    

    return train_gen, test_gen


# In[ ]:


# Hyperparameters

img_dims = 200
epochs = 10
batch_size = 20


# In[ ]:


# Getting the data
train_gen, test_gen = process_data(img_dims, batch_size)


# **WHAT IS TRANFER LEARNING ?**
# Transfer learning refers to a technique for predictive modeling on a different but somehow similar problem that can then be reused partly or wholly to accelerate the training and improve the performance of a model on the problem of interest.
# 
# In deep learning, this means reusing the weights in one or more layers from a pre-trained network model in a new model and either keeping the weights fixed, fine tuning them, or adapting the weights entirely when training the model.
# 
# **WHAT IS INCEPTION V3?**
# Check out the following link by the google ai on Inception V3.
# https://ai.googleblog.com/2016/03/train-your-own-image-classifier-with.html

# In[ ]:


#Inception V3 
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(200,200,3),include_top=False,weights='imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()


# In[ ]:


# Fully Connected Layer
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
Dl_1 = tf.keras.layers.Dropout(rate = 0.2)
pre_prediction_layer = tf.keras.layers.Dense(180, activation='relu')
Dl_2 = tf.keras.layers.Dropout(rate = 0.2)
prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')


model_V3 = tf.keras.Sequential([
  pre_trained_model,
  global_average_layer,
  Dl_1,
  pre_prediction_layer,
  Dl_2,
  prediction_layer
])


# In[ ]:


#Compiling Fully Connected Layer
model_V3.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model_V3.summary()


# In[ ]:


# I will be using the following to reduce the learning rate by the factor of 0.2 when the 'val_loss' will increase in consecutive 3 epochs.
# Callbacks 
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=2, mode='max')


# In[ ]:


#Fitting the model
hist = model_V3.fit_generator(
           train_gen, steps_per_epoch=50, 
           epochs=10, validation_data=test_gen, 
           validation_steps=12 , callbacks=[lr_reduce])


# As seen from above **Training loss (0.5087)** is just above the **Validation loss(0.5068**) model is working quite extraordinary .Considering the powerful nature of the inception v3 , it has accurately guessed  almost all the samples of the validation set at the end. Seems that validation set has simplier examples than the training set .
