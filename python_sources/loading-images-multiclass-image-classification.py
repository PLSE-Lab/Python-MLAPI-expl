#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### ****Loading Images from the directories along with their labels is an important step in any image classification problem. Pre-processing the image data with Augmentation,downsampling,rescaling,Train-Test split and finally feeding it to a CNN can be a tedious task and often gets cumbersome.This kernel trys to breakdown the entire process to simple steps to help us get started with image classification.****
# ### This kernel uses a small MultiClass image dataset with 10 Classes (sub-directories as class tags)

# ### Import required Libraries

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import ImageDataGenerator for Loading Images
from keras.preprocessing.image import ImageDataGenerator


# We use ImageDataGenerator and Flow_from_directory functions which are part of keras preprocessing toolkit to Load the data.
# Details about these classes can be found [here](http://keras.io/preprocessing/image/)

# In[ ]:


# We specify image augmentation parameters as the arguments
# Train - test/validation split can be done with the argument - validation_split
datagen = ImageDataGenerator(rescale=1./255,
                            validation_split = 0.1,rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
# flow_from_directory gets label for an image from the sub-directory it is placed in
# Generate Train data
traingenerator = datagen.flow_from_directory(
        '../input/10_categories-1563192636507/10_categories',
        target_size=(75, 75),
        batch_size=3350,
        subset='training',
        class_mode='categorical')

# Generate Validation data
valgenerator = datagen.flow_from_directory(
        '../input/10_categories-1563192636507/10_categories',
        target_size=(75, 75),
        batch_size=360,
        subset='validation',
        class_mode='categorical')


# ### Split the data into Train and Test
# #### X for the image and y for its corresponding label (This step can be time consuming as this is where the iterator is initiated to load images)

# In[ ]:


x_train,y_train = next(traingenerator)

x_test,y_test = next(valgenerator)


# Plot few random images to check

# In[ ]:


plt.figure(figsize=(20,10))
for i in range(6):
    plt.subplot(1,6,i+1)
    plt.imshow(x_train[i])


# ## These can be further loaded into convnet models for classification
