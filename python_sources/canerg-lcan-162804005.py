#!/usr/bin/env python
# coding: utf-8

# In[105]:


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

# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img


# **Our data is located in three folders:**
# train= contains the training data/images for teaching our model.
# val= contains images which we will use to validate our model. The purpose of this data set is to prevent our model from Overfitting. Overfitting is when your model gets a little too comofortable with the training data and can't handle data it hasn't see....too well.
# test = this contains the data that we use to test the model once it has learned the relationships between the images and their label (Ball/Screw)

# In[106]:


mainDIR = os.listdir('../input/database/database')
print(mainDIR)


# In[107]:


train_folder= '../input/database/database/train/'
val_folder = '../input/database/database/val/'
test_folder = '../input/database/database/test/'


# **Let's set up the training and testing folders.**

# In[108]:


# train 
os.listdir(train_folder)
train_ball = train_folder+'ball/'
train_screw = train_folder+'screw/'


# As a sanity check, let's count how many pictures we have in each training split (train/validation/test):

# In[109]:


print('total training ball images:', len(os.listdir(train_ball)))


# In[110]:


print('total training screw images:', len(os.listdir(train_screw)))


# **DATASET;**
# 
# There are 120 photos in total. {60 ball,60 screw}
# 
# Training photos are 80 photos in total. {40 ball,40 screw}
# 
# Validation photos are 20 photos in total {10 ball,10 screw}
# 
# Test photos are 20 photos in total {10 ball,10 screw}
# 

# ** Lets build our network**

# In[111]:


from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[112]:


model.summary()


# For our compilation step, we'll go with the RMSprop optimizer as usual. Since we ended our network with a single sigmoid unit, we will use binary_crossentropy.

# In[113]:


from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# **Data preprocessing**
# 
# 
# As you already know by now, data should be formatted into appropriately pre-processed floating point tensors before being fed into our network. Currently, our data sits on a drive as JPEG files, so the steps for getting it into our network are roughly:
# 
# * Read the picture files.
# * Decode the JPEG content to RBG grids of pixels.
# * Convert these into floating point tensors.
# * Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).

# In[114]:


from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_folder,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=5,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_folder,
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary')


# In[115]:


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# In[116]:


history = model.fit_generator(
      train_generator,
      steps_per_epoch=16,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=4)


# **Batch size is 5 ,So 1 Epoch completion 80/5=16; in case validation 20/5=4 will be.**
# 

# In[117]:


model.save('ball_screw_small_1.h5')


# In[118]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[119]:


from keras import layers
from keras import models
from keras.regularizers import l2


model2= models.Sequential()
model2.add(layers.Conv2D(32,(3, 3),kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),activation='relu',
                        input_shape=(150, 150, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3),kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3),kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3),kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512,kernel_regularizer=l2(0.0000001), bias_regularizer=l2(0.0000001),activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))


# The Convolutional layers (Conv2D) also use the kernel_regularizer and bias_regularizer arguments to define a regularizer.

# In[120]:


from keras import optimizers

model2.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# In[121]:


history2= model2.fit_generator(
      train_generator,
      steps_per_epoch=16,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=4)


# In[122]:


import matplotlib.pyplot as plt


original_model_loss= history.history['val_loss']
l2_regularized_model_loss = history2.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, original_model_loss, 'b', label='Original model loss')
plt.plot(epochs, l2_regularized_model_loss, 'r', label='l2_regularized_model loss')
plt.title('Original Model and l2_regularized_model Loss')
plt.legend()

plt.figure()


plt.show()


# > Using data augmentation
# 
# 
# Overfitting is caused by having too few samples to learn from, rendering us unable to train a model able to generalize to new data. Given infinite data, our model would be exposed to every possible aspect of the data distribution at hand: we would never overfit. Data augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number of random transformations that yield believable-looking images. The goal is that at training time, our model would never see the exact same picture twice. This helps the model get exposed to more aspects of the data and generalize better.
# 
# In Keras, this can be done by configuring a number of random transformations to be performed on the images read by our ImageDataGenerator instance. Let's get started with an example:

# In[123]:


datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# These are just a few of the options available (for more, see the Keras documentation). Let's quickly go over what we just wrote:
# 
# **rotation_range** is a value in degrees (0-180), a range within which to randomly rotate pictures.
# 
# **width_shift** and **height_shift** are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
# 
# **shear_range** is for randomly applying shearing transformations.
# 
# **zoom_range** is for randomly zooming inside pictures.
# 
# **horizontal_flip** is for randomly flipping half of the images horizontally -- relevant when there are no assumptions of horizontal asymmetry (e.g. real-world pictures).
# 
# **fill_mode** is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
# Let's take a look at our augmented images:

# In[124]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# Let's train our network using data augmentation and dropout

# In[125]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_folder,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=5,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_folder,
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=16,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=4)


# Let's save our model -- we will be using it in the section on convnet visualization.

# In[126]:


model.save('ball_and_screw_small_2.h5')


# In[127]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Thanks to data augmentation and dropout, we are no longer overfitting: the training curves are rather closely tracking the validation curves.
# 
# By leveraging regularization techniques even further and by tuning the network's parameters (such as the number of filters per convolution layer, or the number of layers in the network), we may be able to get an even better accuracy, likely up.

# In[ ]:




