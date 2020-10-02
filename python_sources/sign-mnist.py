#!/usr/bin/env python
# coding: utf-8

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import skimage.data
import keras
import skimage.data
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate
from keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[23]:


os.listdir("../input")


# In[24]:


path = "../input/sign_mnist_"


# In[25]:


def load_data():
    df = pd.read_csv(path + "train.csv")
    y = df.iloc[:,0].values
    x = df.iloc[:,1:].values
    x = np.reshape(x, (x.shape[0],28,28,1))
    df = pd.read_csv(path + "test.csv")
    y2 = df.iloc[:,0].values
    x2 = np.array(df.iloc[:,1:].values)
    x2 = np.reshape(x2, (x2.shape[0],28,28,1))
    x = x / 255
    x2 = x2 / 255
    return (x,y),(x2,y2)    


# In[26]:


(x_train,y_train),(x_test,y_test)=load_data()


# In[27]:


num_classes = 25
L = 10
num_filter = 20


# In[28]:


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[29]:


def dense_block(input, filter_size=(3,3)):
  temp = input
  for l in range(L):
    B = BatchNormalization()(temp)
    A = Activation('relu')(B)
    conv = Conv2D(num_filter, kernel_size=filter_size, padding='same', use_bias=False)(A)
    conc = Concatenate(axis=-1)([temp,conv])
    temp = conc
  return temp


# In[30]:


def trans_block(input):
  B2 = BatchNormalization()(input)
  conv2 = Conv2D(num_filter, (1,1), use_bias=False, padding='same')(B2)
  avg = AveragePooling2D(pool_size=(2,2))(conv2)
  return avg


# In[31]:


def last_layer(input):
  B3 = BatchNormalization()(input)
  avg2 = AveragePooling2D(pool_size=(2,2))(B3)
  F = Flatten()(avg2)
  out = Dense(num_classes, activation='softmax')(F)
  return out


# In[32]:


#model

inputs = Input(shape=(x_train[0].shape))
x = Conv2D(num_filter, (3,3), padding='same', use_bias=False)(inputs)

#block 1
x = dense_block(x)
x = trans_block(x)

#block 2
x = dense_block(x)
x = trans_block(x)

#block 3
#x = dense_block(x)
#x = trans_block(x)

#last block
x = dense_block(x)
output = last_layer(x)


# In[33]:


model = Model(inputs=inputs, outputs=output)


# In[40]:


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[41]:


model.summary()


# In[ ]:


model.fit(x_train,y_train, batch_size=128, epochs=2, validation_data=(x_test,y_test))


# In[ ]:




