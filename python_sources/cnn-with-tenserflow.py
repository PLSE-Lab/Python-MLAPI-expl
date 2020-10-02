#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.shape


# In[ ]:


train = np.array(train)


# In[ ]:


xtrain = np.zeros((train.shape[0],28,28))
ytrain = np.zeros(train.shape[0])
for i in range(train.shape[0]):
    ytrain[i] = train[i][0]
    xtrain[i] = train[i][1:].reshape((28,28))


# In[ ]:


test = np.array(test)
xtest = np.zeros((test.shape[0],28,28))
for i in range(test.shape[0]):
    xtest[i] = test[i].reshape((28,28))


# In[ ]:


xtrain = xtrain.reshape((xtrain.shape[0],28,28,1))
xtrain= xtrain/255
xtest = xtest.reshape((xtest.shape[0],28,28,1))
xtest= xtest/255


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])


# In[ ]:


model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(xtrain,ytrain,epochs=10)


# In[ ]:


for i in range(xtest.shape[0]):
    print(i+1,np.argmax(model.predict(xtest[i].reshape((1,28,28,1)))))


# In[ ]:




