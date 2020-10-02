#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


test_data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
test_label=test_data['label']
test_data=test_data.drop('label',axis=1)
test_data.head()


# In[ ]:


train_data=pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')
train_label=train_data['label']
train_data=train_data.drop('label',axis=1)
train_data.head()


# In[ ]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9),metrics=['acc'])


# In[ ]:


model.fit(train_data.to_numpy(),train_label.to_numpy(),epochs=5)


# In[ ]:




