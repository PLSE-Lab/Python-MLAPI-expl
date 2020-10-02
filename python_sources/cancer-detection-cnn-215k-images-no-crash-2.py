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


# In[ ]:


import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from skimage.io import imread
from skimage.io import imshow

import os


# In[ ]:


train = pd.read_csv("../input/train_labels.csv")


# In[ ]:


train.head()


# In[ ]:


print("Number of training smaples -->" ,len(train))


# In[ ]:


# Function to generate full path of image file

def train_func_image_file(x):
    folder = '../input/train/'
    path = folder + x + '.tif'
    return path


# In[ ]:


# Create image path column in frame

train['path'] = train['id'].apply(train_func_image_file)


# In[ ]:


print(train['path'][0])


# In[ ]:


# Read image file using skimage imread functionality
# Loading all training samples might blow off kernel due to limited memory , so taking maximum possible data

train['image'] = train['path'][0:215000].map(imread)


# In[ ]:


print(imshow(train['image'][1]))


# In[ ]:


# Function to crop image , to reduce memory usage but maintaining target area of image 30x30

def crop(x):
    return x[24:72, 24:72]


# In[ ]:


# Create new column for image crop

train['image_crop'] = train['image'][0:215000].map(crop)


# In[ ]:


print("Cropped image" ,imshow(train['image_crop'][1]))


# In[ ]:


print("Dimension of image --->" ,train['image'][0].shape)


# In[ ]:


print("Dimension of crop image --->" ,train['image_crop'][0].shape)


# In[ ]:


# Drop unwanted columns to release space
train = train.drop(['path'], axis=1)


# In[ ]:


train = train.drop(['image'], axis=1)


# In[ ]:


# Garbage collector to release memory

import gc; 
gc.collect()


# In[ ]:


# Create training array for individual image

x_train = np.stack(list(train.image_crop.iloc[0:215000]), axis = 0)


# In[ ]:


train = train.drop(['image_crop'], axis=1)


# In[ ]:


import gc; 
gc.collect()


# In[ ]:


x_train = x_train.astype('float32')


# In[ ]:


# Normalise array values

x_train /= 255


# In[ ]:


# Label is the target variable

y_train = train['label'][0:215000]


# In[ ]:


del train


# In[ ]:


import gc; 
gc.collect()


# In[ ]:


# Neural network variables to be used

img_rows, img_cols = 48, 48

input_shape = (img_rows, img_cols, 3)

batch_size = 128
epochs = 4


# In[ ]:


# Neural network with multiple layers

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


# Train model

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)


# In[ ]:


del x_train


# In[ ]:


import gc; 
gc.collect()


# In[ ]:


# Create list of test image files

image_file = []
for file in os.listdir("../input/test/"):
    image_file.append(file)


# In[ ]:


# Create test data frame

test = pd.DataFrame(image_file,columns=['file'])


# In[ ]:


test.head()


# In[ ]:


# Function to generate image test file

def test_func_image_file(x):
    folder = '../input/test/'
    path = folder + x
    return path


# In[ ]:


test['path'] = test['file'].apply(test_func_image_file)


# In[ ]:


# Test data image processing

test['image'] = test['path'][0:].map(imread)


# In[ ]:


test['image_crop'] = test['image'][0:].map(crop)


# In[ ]:


test = test.drop(['image'], axis=1)


# In[ ]:


x_test = np.stack(list(test.image_crop.iloc[0:]), axis = 0)


# In[ ]:


test = test.drop(['image_crop'], axis=1)


# In[ ]:


import gc; 
gc.collect()


# In[ ]:


x_test = x_test.astype('float32')


# In[ ]:


x_test /= 255


# In[ ]:


test['id'] = test['file'].apply(lambda x: os.path.splitext(x)[0])


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


predictions = predictions.reshape(len(x_test),)


# In[ ]:


predictions = (predictions > 0.5).astype(np.int)


# In[ ]:


test['label'] = pd.Series(predictions)


# In[ ]:


print("Cancer Detected - True Positive --> ",len(test['label'][test['label']==1]))


# In[ ]:


print("NO Cancer Detected - True Negative --> ",len(test['label'][test['label']==0]))


# In[ ]:


test = test.drop(['file','path'], axis=1)


# In[ ]:


test.head()


# In[ ]:


test.to_csv("submission.csv", columns = test.columns, index=False)


# In[ ]:




