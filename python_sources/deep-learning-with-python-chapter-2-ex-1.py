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


from keras.datasets import mnist


# In[ ]:


Data = np.load("../input/mnist.npz")


# In[ ]:


Data.keys()


# In[ ]:


X_train = Data['x_train']
X_test = Data['x_test']
y_train = Data['y_train']
y_test = Data['y_test']


# In[ ]:


from keras import models
from keras import layers


# In[ ]:


model = models.Sequential()


# In[ ]:


model.add(layers.Dense(512, activation='relu', input_shape = (28*28, )))


# In[ ]:


model.add(layers.Dense(10, activation='softmax'))


# In[ ]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


X_train = X_train.reshape((60000,28*28))
X_train = X_train.astype('float32')/255


# In[ ]:


X_test = X_test.reshape((10000,28*28))
X_test = X_test.astype('float32')/255


# In[ ]:


from keras.utils import to_categorical


# In[ ]:


y_train = to_categorical(y_train)


# In[ ]:


y_test = to_categorical(y_test)


# In[ ]:


model.fit(X_train, y_train, epochs=5, batch_size=128)


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test)


# In[ ]:


test_acc


# In[ ]:




