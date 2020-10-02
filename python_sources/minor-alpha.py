#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# import all libraries
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import keras
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# In[3]:


data = pd.read_csv("../input/mnist_train.csv")
data.head()


# In[4]:


X = data.drop(["5"], axis = 1)
y = data["5"]
X.shape


# In[5]:


X_dup = X.values.astype("float32").reshape(X.shape[0], 28, 28)
y_dup = y.values.astype("int32")
X.shape


# In[6]:


# X_dup = X_dup.reshape(X.shape[0], 28, 28)


# In[7]:


for index in range(8, 11):
    plt.subplot(330 + (index - 1))
    plt.imshow(X_dup[index], cmap = plt.get_cmap("gray"))
    plt.title(y_dup[index])


# In[8]:


X_dup = X_dup.reshape(X_dup.shape[0], 28, 28, 1)

X_train, X_test, y_train, y_test = train_test_split(X_dup, y_dup, test_size = 0.3)


# In[9]:


X_train_mean = X_train.mean().astype(np.float32)
X_train_std = X_train.std().astype(np.float32)


# In[10]:


X_train = (X_train - X_train_mean)/X_train_std


# In[11]:


# Label Encoding
y_train = to_categorical(y_train, num_classes =10)
y_test = to_categorical(y_test, num_classes = 10)


# In[12]:


# create CNN model for layers
input_shape = (28, 28, 1)
num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "Same", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(125, activation = "relu"))
model.add(Dropout(0.30))
model.add(Dense(10, activation = "softmax"))
model.summary()


# In[13]:


optimizer = Adam(lr = .001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = 0, amsgrad = False)


# In[14]:


# Compile the model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

# learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = .5,
                                            min_lr = .00001)


# In[15]:


# Data Augmentation
datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False, 
                            featurewise_std_normalization = False, samplewise_std_normalization = False,
                            zca_whitening = False, rotation_range = 10, zoom_range = .1, 
                            width_shift_range = .1, height_shift_range = .1, horizontal_flip = False, 
                            vertical_flip = False)
datagen.fit(X_train)


# In[16]:


# Fitting the model
epochs = 3
batch_size = 100

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), epochs = epochs, 
                             verbose = 1, steps_per_epoch = X_train.shape[0]//batch_size, 
                              callbacks = [learning_rate_reduction])


# In[17]:


# Saving the model
model.save("model.h5")


# In[ ]:




