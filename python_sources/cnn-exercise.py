#!/usr/bin/env python
# coding: utf-8

# # CNN Exercise with Fashion MNIST Data
# 
# 1. [Importing Libraries and Stuff](#1)
# 1. [A Quick Look on the Data](#2)
# 1. [Getiing Data Ready](#3)
# 1. [Creating Model](#4)
# 1. [Prediction and Result](#5)

# <a id = '1' ></a>
# ## Importing Libraries and Stuff

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from imgaug import augmenters
from random import randint


# In[ ]:


train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")


# <a id = '2' ></a>
# ## A Quick Look on the Data

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# <a id = '3' ></a>
# ## Getting Data Ready

# In[ ]:


rows, cols = 28, 28
input_shape = (rows, cols, 1)


# In[ ]:


X = np.array(train.iloc[:, 1:])
y = to_categorical(np.array(train.iloc[:, 0]))


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 36)


# In[ ]:


x_test = np.array(test.iloc[:, 1:])
y_test = to_categorical(np.array(test.iloc[:, 0]))


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
x_val = x_val.reshape(x_val.shape[0], rows, cols, 1)


# In[ ]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')


# In[ ]:


x_train = x_train/255
x_test = x_test/255
x_val = x_val/255


# <a id = '4' ></a>
# ## Creating Model

# In[ ]:


batch_size = 64
epochs = 100
num_classes = 10

img_rows, img_cols = 28, 28

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape = input_shape))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = categorical_crossentropy,
              optimizer = Adam(),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=0)


# In[ ]:


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=1,
          validation_data=(x_val, y_val))
model.evaluate(x_test, y_test, verbose=0)


# <a id = '5' ></a>
# ## Prediction and Result

# In[ ]:


prediction = model.predict_classes(x_test)

