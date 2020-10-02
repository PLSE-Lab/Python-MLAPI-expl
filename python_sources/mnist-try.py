#!/usr/bin/env python
# coding: utf-8

# First attempt on MNIST dataset using Kaggle Notebook
# 

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


# # Importing Libraries and Data

# In[ ]:


train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D,Flatten,MaxPool2D,Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


y_train = train['label']
y_train.value_counts()


# Converting y_train to categorical data

# In[ ]:


y_train = to_categorical(y_train,num_classes=10)


# In[ ]:


X_train = train.drop(columns=['label'])


# Scaling the X_train and test to be in range [0,1],grayscale
# 

# In[ ]:


X_train.head()
X_train = X_train/255.0
test = test/255.0


# **RESHAPING THE X_Train to have dimension (28,28,1) where 1 is denotes the channel**

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# # Model 

# In[ ]:


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer=keras.optimizers.SGD(lr=0.01,decay=10e-6),metrics=['accuracy'],loss='categorical_crossentropy')


# In[ ]:


history = model.fit(x=X_train,y=y_train,epochs=10,validation_split=0.2)


# In[ ]:


hist = history.history
hist.keys()


# In[ ]:


acc = hist['accuracy']
val_loss = hist['val_loss']
loss = hist['loss']
val_acc = hist['val_accuracy']


# **Plotting Training Acc vs Validation Acc
#   Plotting Training Loss vs Validation Loss**

# In[ ]:


epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training Accuracy')
plt.plot(epochs,val_acc,'b',label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.figure()
plt.show()


# In[ ]:


plt.plot(epochs,loss,'r',label='Training Loss')
plt.plot(epochs,val_loss,'bo',label='Validation Loss')
plt.title('Training vs Validation loss')
plt.legend()
plt.figure()
plt.show()


# In[ ]:


result = model.predict(test)


# In[ ]:


result = np.argmax(result,axis=1)


# In[ ]:


result


# In[ ]:


result = pd.Series(result,name="Label")


# In[ ]:


result


# In[ ]:


final = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)


# In[ ]:


final.head()


# In[ ]:


final.to_csv('MNIST_attempt-1.csv')

