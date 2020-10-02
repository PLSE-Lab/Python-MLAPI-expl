#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import sys
import array
import cv2
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **Converting the Train data into Pixels**

# In[ ]:


columnNames = list()
columnNames.append('label')
for i in range(784):
    pixel = 'pixel'
    pixel += str(i)
    columnNames.append(pixel)
train_data = pd.DataFrame(columns = columnNames)
i=0
count=1
for dirname, _, filenames in os.walk('/kaggle/input/devnagri-handwritten-character/DEVNAGARI_NEW/TRAIN'):
    for filename in filenames:
        image_name = os.path.join(dirname, filename)
        #print(os.path.join(dirname, filename))
        label = dirname[dirname.rindex('/')+1:]
        #print(label)
        img = Image.open(image_name)
        rawData = img.load()
        #print(rawData)
        data = []
        data.append(label)
        for y in range(28):
            for x in range(28):
                data.append(rawData[x,y])
        k = 0
        train_data.loc[i] = [data[k] for k in range(785)]
        i = i+1
    print(count)
    count = count + 1


# In[ ]:


train_data.head()


# # **Converting the Test data into Pixels**

# In[ ]:


columnNames = list()
columnNames.append('label')
for i in range(784):
    pixel = 'pixel'
    pixel += str(i)
    columnNames.append(pixel)
test_data = pd.DataFrame(columns = columnNames)
i=0
count=1
for dirname, _, filenames in os.walk('/kaggle/input/devnagri-handwritten-character/DEVNAGARI_NEW/TEST'):
    for filename in filenames:
        image_name = os.path.join(dirname, filename)
        #print(os.path.join(dirname, filename))
        label = dirname[dirname.rindex('/')+1:]
        #print(label)
        img = Image.open(image_name)
        rawData = img.load()
        #print(rawData)
        data = []
        data.append(label)
        for y in range(28):
            for x in range(28):
                data.append(rawData[x,y])
        k = 0
        test_data.loc[i] = [data[k] for k in range(785)]
        i = i+1
    print(count)
    count = count + 1


# In[ ]:


test_data.to_csv("test.csv",index=False)
train_data.to_csv('train.csv',index=False)
train = train_data
test = test_data


# **PREPARE DATA FOR NEURAL NETWORK**
# 
# Separating the labels from the training data and converting the train and test data to a format which is suitable to be fed to the CNN

# In[ ]:


Y_train = train["label"]
Y_train = Y_train.astype(int)
Y_train = Y_train - 1
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test.drop(labels = ["label"],axis = 1)
X_test = X_test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 48)


# In[ ]:


datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(48, activation='softmax'))

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)


# In[ ]:


history = model.fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = 60, steps_per_epoch = X_train2.shape[0]//64)


# In[ ]:




