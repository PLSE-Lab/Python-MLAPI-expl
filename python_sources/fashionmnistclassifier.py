#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import DepthwiseConv2D

print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv('../input/fashion-mnist_train.csv')
df_test = pd.read_csv('../input/fashion-mnist_test.csv')


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


y_train = df_train['label']
X_train = df_train.drop(columns=['label'])
y_test = df_test['label']
X_test = df_test.drop(columns=['label'])


# In[ ]:


## Normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


## One hot encode
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# In[ ]:


## Basic info about the data
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)


# #### Thus multiclass classification problem
# 
# ## Fully Connected Network

# In[ ]:


def model():
    model = Sequential()
    model.add(Dense(784, activation='relu', input_dim=784))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


# In[ ]:


model = model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


earlyStopper = EarlyStopping(monitor='acc', patience=1)


# ##### Initial trial with Densenet

# In[ ]:


my_hist = model.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])


# In[ ]:


eval = model.evaluate(x=X_test, y=y_test, batch_size=100)


# In[ ]:


eval


# In[ ]:


plt.plot(my_hist.history['acc'])
plt.plot(my_hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc='right')
plt.title('accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.show()


# ## Standard CNN

# In[ ]:


X_train = X_train.reshape((X_train.shape[0], 28,28,1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))


# In[ ]:


def modelCNN():
    model1 = Sequential()
    model1.add(Conv2D(5, kernel_size=[3,3], padding='valid', input_shape=(28,28,1)))
    model1.add(Conv2D(25, kernel_size=[5,5], padding='valid', activation='relu'))
    model1.add(MaxPool2D(pool_size=[3,3]))
    model1.add(Conv2D(50, kernel_size=[3,3], padding='same', activation='relu'))
#     model1.add(MaxPool2D(pool_size=[3,3]))
    model1.add(Conv2D(100, kernel_size=[3,3], padding='valid', activation='relu'))
    model1.add(Flatten())
    model1.add(Dense(1024, activation='relu'))
    model1.add(Dense(512, activation='relu'))
    model1.add(Dense(10, activation='softmax'))
    return model1


# In[ ]:


model1 = modelCNN()


# In[ ]:


model1.summary()


# In[ ]:


model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


hist = model1.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])


# In[ ]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc='right')
plt.title('accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.show()


# ## Depthwise Convolution

# In[ ]:


def modelDepthWise():
    model2 = Sequential()
    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='valid', depth_multiplier=5, input_shape=(28,28,1)))
    model2.add(DepthwiseConv2D(kernel_size=[5,5], padding='valid', depth_multiplier=5, activation='relu'))
    model2.add(MaxPool2D(pool_size=[3,3]))
    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='same', depth_multiplier=2, activation='relu'))
    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='valid', depth_multiplier=2, activation='relu'))
    model2.add(Flatten())
    model2.add(Dense(1024, activation='relu'))
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(10, activation='softmax'))
    return model2


# In[ ]:


modelDepthwise = modelDepthWise()


# In[ ]:


modelDepthwise.summary()


# In[ ]:


modelDepthwise.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


hist = modelDepthwise.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])


# In[ ]:


plt.plot(hist.history['acc'])
plt.plot(hist.history['loss'])
plt.legend(['accuracy', 'loss'], loc='right')
plt.title('accuracy and loss')
plt.xlabel('epoch')
plt.ylabel('accuracy/loss')
plt.show()

