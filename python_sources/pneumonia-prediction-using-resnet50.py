#!/usr/bin/env python
# coding: utf-8

# **Overview**
# 
# Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it.
# The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe.
# There are several types of infectious agents that can cause pneumonia.
# * Bacterial pneumonia
# * Viral pneumonia
# * Fungal pneumonia
# 
# **Pneumonia diagnosis**
# 
# Depending on the severity of your symptoms and your risk for complications, your doctor may also order one or more of these tests:
# * Chest X-ray
# * Blood culture
# * Sputum culture
# * Pulse oximetry
# * CT scan
# * Fluid sample
# * Bronchoscopy

# In[ ]:


get_ipython().system('pip install Augmentor')


# In[ ]:


# Importing necessary libraries

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import cv2
import Augmentor
from keras.applications.resnet50 import ResNet50

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


# Input data files are available in the "../input/chest-xray-pneumonia/chest_xray" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

directory = os.listdir('../input/chest-xray-pneumonia/chest_xray')
print(directory)


# The dataset is divided into three sets: 1) train set 2) validation set and 3) test set. Let's grab the dataset

# In[ ]:


data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

# Path to train directory 
train_dir = data_dir/'train'

# Path to validation directory
val_dir = data_dir/'val'

# Path to test directory
test_dir = data_dir/'test'
print(train_dir)


# Each of the above directory contains two sub-directories:
# * NORMAL: These are the samples that describe the normal (no pneumonia) case.
# * PNEUMONIA: This directory contains those samples that are the pneumonia cases.

# In[ ]:


# Get the list of all the images
normal_train_cases = train_dir.glob('NORMAL/*.jpeg')
pneumonia_train_cases = train_dir.glob('PNEUMONIA/*.jpeg')


# In[ ]:


train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in normal_train_cases:
    train_data.append((img,0))
    
# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_train_cases:
    train_data.append((img, 1))
    
# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)


# In[ ]:


train_data.head()


# In[ ]:


train_data.tail()


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='label', data=train_data, palette='RdBu_r')


# Checking wether images are Grey-Scale or RGB

# In[ ]:


resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('../input/chest-xray-pneumonia/chest_xray/train/*')

model = Sequential()

model.add(resnet)

model.add(Flatten())

model.add(Dense(len(folders), activation='softmax'))

model.summary()

# Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[ ]:


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = '../input/chest-xray-pneumonia/chest_xray/train/'
HEIGHT = 256
WIDTH = 256
BATCH_SIZE = 32

from imblearn.keras import balanced_batch_generator

# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

datagen = ImageDataGenerator()
balanced_gen = BalancedDataGenerator(x, y, datagen, batch_size=32)


training_set = train_datagen.flow_from_directory(TRAIN_DIR',
                                                 target_size = (HEIGHT, WIDTH),
                                                 batch_size = BATCH_SIZE,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/test/',
                                            target_size = (HEIGHT, WIDTH),
                                            batch_size = BATCH_SIZE,
                                           class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch=len(training_set),
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=len(test_set))


# In[ ]:




