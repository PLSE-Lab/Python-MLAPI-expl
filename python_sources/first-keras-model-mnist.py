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


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test=pd.read_csv('../input/test.csv')


# In[ ]:


train_y = train.label
train.drop(["label"], axis = 1, inplace = True)
train_x = train


# In[ ]:


#normalisation here
import matplotlib.pyplot as plt
train_x = train_x / 255.0
train_x.shape


# In[ ]:


train_x = train_x.values.reshape(-1,28,28,1) #reshaping
train_x=to_categorical(x_train)


# In[ ]:


#let's see some numbers
import matplotlib.pyplot as plt
plt.imshow(train_x[1][:,:,0],cmap='gray')

plt.show()
plt.imshow(train_x[8][:,:,0],cmap='gray')
plt.show()


# In[ ]:


#let's create simple deep learning model
import tensorflow as tf 

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    
    
])


# In[ ]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics = ['accuracy'])


# In[ ]:


#here I trained it for a while, it was improving all the time 
#but finally network gave us 0.96814 
model.fit(train_x, train_y, epochs=1, steps_per_epoch=5)


# In[ ]:


test = test.values.reshape(-1,28,28,1)
test = test / 255.0


# In[ ]:


#now let's try CNN :) If it's better then we shoud go very close to 100%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])


# In[ ]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(train_x,train_y, epochs=11, steps_per_epoch=70)


# In[ ]:


pred = model.predict(test)
pred[44]


# In[ ]:


pred = np.argmax(pred, axis=1) # we need only the best fits 


# In[ ]:


pred = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':pred })


# In[ ]:


pred.to_csv("predi.csv",index=False)


# In[ ]:





# In[ ]:




