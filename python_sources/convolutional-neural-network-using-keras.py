#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/fruits-360_dataset/fruits-360"))

# Any results you write to the current directory are saved as output.

# prepare image directory and labels
train_dir='../input/fruits-360_dataset/fruits-360/Training'
test_dir='../input/fruits-360_dataset/fruits-360/Test'
test_labels=os.listdir(test_dir)
labels=os.listdir(train_dir)
print(labels)
print(test_labels)

# Make Convolutional Neural Network
model=Sequential()
model.add(Conv2D(3,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(len(labels),activation='softmax'))
model.summary()

# Preprocessing image
train_gen=ImageDataGenerator().flow_from_directory(train_dir,batch_size=32,classes=labels,target_size=(64,64))
validation_gen=ImageDataGenerator().flow_from_directory(test_dir,batch_size=32,classes=labels,target_size=(64,64))

# Compile and fit

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(train_gen,steps_per_epoch=1000,epochs=10,validation_steps=400,validation_data=validation_gen)


# In[ ]:




