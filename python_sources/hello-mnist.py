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


# # Read Data

# In[2]:


df_test = pd.read_csv('../input/test.csv')


# In[3]:


df_train = pd.read_csv('../input/train.csv')


# In[4]:


(X_train, y_train) = df_train.drop('label', axis=1), df_train['label']


# In[5]:


X_train.head(), y_train.head()


# In[6]:


X_train.shape, y_train.shape


# In[7]:


X_test = pd.read_csv('../input/test.csv')

df_test['label'] = 0
y_test = df_test['label']


# In[8]:


X_test.head(), y_test.head()


# In[9]:


X_test.shape, y_test.shape


# # Data Prep

# In[10]:


image_height, image_width = 28, 28


# In[11]:


X_train = X_train.values.reshape(42000, image_height * image_width)

print(X_train.shape)


# In[12]:


X_test = X_test.values.reshape(28000, image_height * image_width)

print(X_test.shape)


# In[13]:


# make into float numbers
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# hot encoding from 0 to 1
X_train /= 255.0
X_test /= 255.0

print(X_train[0])


# In[14]:


print(y_train.shape)


# In[ ]:





# In[15]:


from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, 10)
print(y_train.shape)


# In[16]:


y_test = to_categorical(y_test, 10)
print(y_test.shape)


# # Keras Model

# In[17]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


# In[18]:


# first layer
model.add(Dense(512, activation='relu', input_shape=(784,)))

#second layer
model.add(Dense(512, activation='relu'))

# third layer
model.add(Dense(10, activation='softmax'))


# In[19]:


model.compile(optimizer='adam',                 # adam is the got-to optimizer in general
              loss='categorical_crossentropy',  # 10 classes/bins. this function allows for that
              metrics=['accuracy']              # accuracy
             )


# In[20]:


model.summary()


# In[21]:


from keras.callbacks import TensorBoard


# In[22]:


tboard = TensorBoard(log_dir='./output', 
                     histogram_freq=5, 
                     write_graph=True, 
                     write_images=True
                    )


# In[23]:


history = model.fit(X_train, y_train, 
                    epochs=8, 
                    validation_data=(X_test, y_test),
                    validation_split=1/6, 
                    callbacks=[tboard]
                   )


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(history.history['acc'])      # accuracy of training set
plt.plot(history.history['val_acc'])  # accuracy of testing set


# In[25]:


plt.plot(history.history['loss'])     # loss score


# In[26]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[27]:


score = model.evaluate(X_test, y_test)


# In[28]:


score


# In[ ]:




