#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import TensorBoard
from keras.utils import np_utils

import numpy as np
import pandas as pd


# In[ ]:


x_train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/train.csv')

x_train.head()


# In[ ]:


y_train = x_train['label']
x_train = x_train.drop('label',axis=1)
x_train.head()


# In[ ]:


print(x_train.shape)
print(y_train.shape)


# In[ ]:


x_train = x_train.astype('float32')
x_train = x_train/255.
y_train= np_utils.to_categorical(y_train)


# In[ ]:


model = Sequential()

model.add(
    Dense(
        units=64,
        input_shape=(784,),
        activation='relu'
    )
)


# In[ ]:


model.add(
    Dense(
        units=10,
        activation='softmax'
    )
)


# In[ ]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


history_adam = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2
)


# In[ ]:


test_y = test['label']
test_x =test.drop('label',axis=1)
test_x.shape


# In[ ]:


y_pre = model.predict(test_x)
y_pre = y_pre.argmax(axis = 1)
sub = pd.DataFrame({"ImageId":test_y.index + 1,"Label":y_pre})
sub.to_csv('submission.csv',index=False)

