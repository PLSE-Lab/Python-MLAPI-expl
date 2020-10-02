#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data = []
for i in os.listdir('../input/histopathologic-cancer-detection/train'):
    img = cv2.imread(os.path.join('../input/histopathologic-cancer-detection/train',i))
    img = np.array(img).reshape(96,96,3)
    train_data.append(img)
x = np.array(train_data)


# In[ ]:


x.shape


# In[ ]:


data = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')


# In[ ]:


y = np.array(data.drop(['id'], axis=1))


# In[ ]:


y.shape


# In[ ]:


print(y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=4, test_size = 0.20)


# In[ ]:


import tensorflow as tf
model = tf.keras.Sequential()
model.add(layers.Conv2D(16, activation="relu", kernel_size=(3, 3),input_shape=(96,96,3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32, activation="relu", kernel_size=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, activation="relu", kernel_size=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, activation="relu", kernel_size=(3, 3)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train,y_train, epochs=1, validation_data=(X_test,y_test), batch_size=32)


# In[ ]:





# In[ ]:




