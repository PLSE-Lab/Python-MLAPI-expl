#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import train_test_split
from skimage import io
import os
import scipy.misc
from scipy.misc import imread, imresize
from keras import regularizers
import csv


# Function to get training and validation images.

# In[2]:


import cv2

def load_images(path):
    img_data = []
    labels = []
    idx_to_label = []
    i = -1
    for fruit in os.listdir(path):
        fruit_path = os.path.join(path,fruit)
        labels.append(fruit)
        i = i+1
        for img in os.listdir(fruit_path):
            img_path = os.path.join(fruit_path,img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_data.append(image)
            idx_to_label.append(i)
    return np.array(img_data),np.array(idx_to_label),labels


# Calling the function.

# In[3]:


trn_data_path = '../input/fruits-360_dataset_2018_06_03/fruits-360/Training'
val_data_path = '../input/fruits-360_dataset_2018_06_03/fruits-360/Validation'
X_train,y_train,label_data = load_images(trn_data_path)
print(X_train.shape)
print(y_train.shape)
print(label_data)
X_test,y_test,label_data_garbage = load_images(val_data_path)


# One hot encoding.

# In[4]:


num_of_classes = 65
Y_train = np_utils.to_categorical(y_train, num_of_classes)
Y_test = np_utils.to_categorical(y_test, num_of_classes)


# Model.
# I played around with the dropout values and achieved better score at 0.7
# activations can be Relu, elu, LeakyRelu.

# In[5]:


model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3 )))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(65))

model.add(Activation('softmax'))


# In[6]:


model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])


# No augmentation.

# In[7]:


model.fit(X_train,Y_train, epochs=1,batch_size = 32,
                    validation_data=(X_test,Y_test))


# In[8]:


model.fit(X_train,Y_train, epochs=1,batch_size = 32,
                    validation_data=(X_test,Y_test))


# In[9]:


score = model.evaluate(X_test, Y_test)
print()
print('Test loss: ', score[0])
print('Test Accuracy', score[1])


# Augmentation.

# In[ ]:


gen = ImageDataGenerator(width_shift_range=.2, 
                             height_shift_range=.2,
                         zoom_range = 0.1,
                        horizontal_flip = 'True')

test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=32)
test_generator = test_gen.flow(X_test, Y_test, batch_size=32)


# In[ ]:


model.fit_generator(train_generator, epochs=5, 
                    validation_data=test_generator)


# In[ ]:




