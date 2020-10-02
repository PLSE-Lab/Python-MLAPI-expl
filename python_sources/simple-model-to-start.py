#!/usr/bin/env python
# coding: utf-8

# ## This is a simple keras model to begin with . I will keep updating it .
# Update 1 -  Playing with number of epochs & thought of combining dig mnist validation data with train data to increase data for training but combining dig mnist is not improving score infact it is reducing the score so  going ahead with only train data . 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


img_rows, img_cols = 28, 28
num_classes = 10


# In[ ]:


def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


# In[ ]:


train_file = "../input/Kannada-MNIST/train.csv"
#dmnist  = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
raw_data = pd.read_csv(train_file)
#raw_data = pd.concat([train,dmnist])


# In[ ]:


raw_data.shape


# In[ ]:


x,y = data_prep(raw_data)


# In[ ]:


y.shape


# In[ ]:


model = Sequential()
model.add(Conv2D(200, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(200, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1280, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


model.fit(x, y,
          batch_size=1000,
          epochs=10,
          validation_split = 0.2)


# In[ ]:


test_file = "../input/Kannada-MNIST/test.csv"
test_data = pd.read_csv(test_file)


# In[ ]:


test_data.shape


# In[ ]:


num_images_t = test_data.shape[0]
tx_as_array = test_data.values[:,1:]
tx_shaped_array = tx_as_array.reshape(num_images_t, img_rows, img_cols, 1)
test_x = tx_shaped_array / 255


# In[ ]:


pred = model.predict_classes(test_x)


# In[ ]:


sub = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")


# In[ ]:


sub.label =pred


# In[ ]:


sub.to_csv("KannadaMnist1.csv",index = False)


# In[ ]:




