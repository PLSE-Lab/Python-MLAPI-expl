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


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_submission_df = pd.read_csv('../input/sample_submission.csv')
print(len(train_df), len(test_df), len(sample_submission_df))


# In[ ]:


import tensorflow as tf


# In[ ]:


train_df.head()


# In[ ]:


import tensorflow.keras as keras 
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


img1 = train_df.iloc[1]

k = img1[1:].values.reshape((28,28))
plt.imshow(k)
plt.show()
print("Label : {}".format(img1[0]))
print(k.shape)


# In[ ]:



msk = np.random.rand(len(train_df)) < 0.8

train_df, val_df = train_df[msk], train_df[~msk]


# In[ ]:


y_train = train_df['label']
y_val = val_df['label']


# In[ ]:


train_df.drop(columns=['label'],inplace=True)
val_df.drop(columns=['label'],inplace=True)


# In[ ]:


train_len, _ = train_df.shape
test_len, _ = test_df.shape
val_len, _ = val_df.shape


# In[ ]:


x_train = train_df.values.reshape((train_len, 28, 28))
x_test = test_df.values.reshape((test_len, 28,28))
x_val = val_df.values.reshape((val_len,28,28))


# In[ ]:


plt.imshow(x_train[0])
plt.show()


# In[ ]:


x_train = x_train.reshape((train_len, 28, 28,1))
x_test = x_test.reshape((test_len, 28,28,1))
x_val = x_val.reshape((val_len, 28, 28, 1))


# In[ ]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)))
model.add(layers.Conv2D(64,3,activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
model.add(layers.Conv2D(64,3,activation='relu',padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[ ]:


model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])


# In[ ]:


get_ipython().run_line_magic('pinfo', 'model.fit')


# In[45]:


model.fit(x_train, y_train, batch_size=64, epochs=25, validation_data=(x_val, y_val))


# In[49]:


y_test = model.predict(x_test, batch_size=64)


# In[57]:


y_test = np.argmax(y_test, axis=1)


# In[58]:


test_df['label'] = y_test


# In[59]:


test_df.head()


# In[66]:


submission = pd.DataFrame({'ImageId': test_df.index, 'Label': y_test})


# In[67]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




