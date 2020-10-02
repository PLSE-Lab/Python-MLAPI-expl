#!/usr/bin/env python
# coding: utf-8

# # MNIST with RNN(LSTM to be exact). 
# 
# In this notebook, I use Long Short-Term Memory(LSTM) to classify the famous MNIST data. If you like it please give me an upvote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')


# In[ ]:


X = np.array(df.drop('label', 1)).reshape(42000, 28,28).astype('float32') / 255
y = np.array(df['label']).astype('float32')


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.imshow(X[100])


# In[ ]:


import tensorflow as tf


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=X.shape[1:], return_sequences=True))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


train_size = round(len(X)*0.8)
X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]
print(len(X_train))
val_size = round(len(X_train)*0.8)
print(val_size)
X_val = X_train[val_size:]
y_val = y_train[val_size:]
X_train = X_train[:val_size]
y_train = y_train[:val_size]


# In[ ]:


model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))


# In[ ]:


accuracy = model.evaluate(X_test, y_test)


# In[ ]:




