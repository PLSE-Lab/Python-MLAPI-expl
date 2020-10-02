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


df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


train = df.select_dtypes(include='number')


# In[ ]:


train


# In[ ]:


Y = train.pop('Yards')


# In[ ]:


Y


# In[ ]:


train


# In[ ]:


import tensorflow as tf


# In[ ]:


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[24])
])


# In[ ]:


my_model.summary()


# In[ ]:


my_model.compile(
    loss = 'mse',
    optimizer = 'adam'
)


# In[ ]:


my_model.fit(train, Y, epochs=10)


# In[ ]:


train = train.dropna()


# In[ ]:


my_model.fit(train, Y, epochs=10)


# In[ ]:


df


# In[ ]:


train = df.select_dtypes(include='number')


# In[ ]:


train.info()


# In[ ]:


train = train.dropna()


# In[ ]:


train.info()


# In[ ]:


Y = train.pop('Yards')


# In[ ]:


train.info()


# In[ ]:


my_model.fit(train, Y, epochs=10)


# In[ ]:


train.pop('GameId')


# In[ ]:


train.pop('PlayId')


# In[ ]:


my_model.fit(train, Y, epochs = 1)


# In[ ]:


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[22])
])

#my_model.compile(
#    loss = 'mse',
#    optimizer = 'adam'
#)


# In[ ]:


my_model.fit(train, Y, epochs = 10)


# In[ ]:


my_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=[22], activation='relu'),
    tf.keras.layers.Dense(1)
])

my_model_2.compile(
    loss = 'mse',
    optimizer = 'adam'
)


# In[ ]:


my_model_2.fit(train, Y, epochs = 10)


# In[ ]:


my_model_2.fit(train, Y, epochs = 10)


# In[ ]:




