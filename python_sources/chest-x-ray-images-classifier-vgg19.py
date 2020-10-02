#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import json
import math
import os
from glob import glob 
from tqdm import tqdm
from PIL import Image
import cv2 # image processing
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization


from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split

from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.applications import VGG16,VGG19
from keras.utils.np_utils import to_categorical
from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img


# In[ ]:


input_path = "../input/chest-xray-pneumonia/chest_xray/"
train_path = input_path +"train/"
test_path = input_path +"test/"
val_path = input_path +"val/"


# In[ ]:


fig, ax = plt.subplots(1,4, figsize=(12,12))
ax = ax.ravel()
plt.tight_layout()
for i, _set in enumerate(['train', 'test']):
    set_path = input_path+_set
    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]),cmap='gray')
    ax[i].set_title('File: {} - Condition: Healthy'.format(_set))
    ax[i+2].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]),cmap='gray')
    ax[i+2].set_title('File: {} - Condition: Infected'.format(_set))


# In[ ]:


# Data Augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')


# In[ ]:


# Create Model with KERAS library
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu",padding="same",input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3,3), activation="relu",padding="same"))
model.add(Conv2D(64, (3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation="relu",padding="same"))
model.add(Conv2D(128, (3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(Conv2D(256, (3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.summary()


# In[ ]:


# Compile Mode
model.compile(loss='binary_crossentropy',
              optimizer= Adam(lr=0.0001),
              metrics=['acc'])


# In[ ]:


# Fit Model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=10
)


# In[ ]:


# model save
model.save_weights("cnn-example-pneumonia-vgg19.h5")


# In[ ]:


# Visualize Loss and Accuracy Rates
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['acc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])


# In[ ]:



