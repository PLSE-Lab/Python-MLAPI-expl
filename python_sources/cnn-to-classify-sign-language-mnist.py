#!/usr/bin/env python
# coding: utf-8

# **Importing the necessary libraries**

# In[80]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.utils import to_categorical 
from keras import backend as K
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Activation

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Data loading**
# 

# Load the data from the csv file which contains label information at 0th column and remaining columns shows the pixel value at each 784 locations.

# In[81]:


df = pd.read_csv('../input/sign_mnist_train.csv')
print(df.shape)
df.head()


# From the dataframe obtain the training data and load the labels. Use to_categorical to convert the labels into one-hot encoding.

# In[82]:


train=df.values[0:,1:]
labels = df.values[0:,0]
labels = to_categorical(labels)
sample = train[1]
plt.imshow(sample.reshape((28,28)))


# **Preparing Train/Test set**

# Normalize the data and convert each row of (1,784) shape into (28,28,1) of the training data.

# In[83]:


print(train.shape,labels.shape)
#normalizing the dataset
train=train/255
train=train.reshape((27455,28,28,1))
plt.imshow(train[1].reshape((28,28)))
print(train.shape,labels.shape)


# **Neural Network model**
# 

# We make use of Convolutional Neural Network(CNN) as our model. Initial layer requires the input shape for each row of our training data which is of the shape (28,28,1) and final layer outputs a 25 dimension output.

# In[84]:


model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (3,3),input_shape = (28,28,1),activation = 'relu',padding = 'same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,kernel_size = (3,3),padding = 'same',activation = 'relu'))
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dense(25,activation = 'softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
h=model.fit(train, labels, validation_split=0.3, epochs=6,batch_size=64)


# Plotting the model accuracy and model loss function values.

# In[85]:


plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('Model accuracy')
plt.show()

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model Loss')
plt.show()


# **Testing**

# For testing we first take a sample from the traning data which in our case is from 4th location which has a label 13 and then need to prepare the data to make it suitable for our model to predict.

# In[92]:


LOC = 25
sample = train[LOC]
plt.imshow(sample.reshape((28,28)))
lbl=labels[LOC]
print(list(lbl).index(1))


# Convert the given image to (1,28,28,1) shape ,normalize it and give it to the model. Find the index of the largest probablitiy from the given set of predictions.

# In[93]:


sample=sample.reshape((1,28,28,1))
res=model.predict(sample)
res=list(res[0])
mx=max(res)
print(res.index(mx))

