#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


def get_data(file):
    data = pd.read_csv(file)
    data = data.to_numpy()
    labels = []
    images = []   
    for row_ind in range(data.shape[0]):
        labels.append(data[row_ind, 0])
        images.append(data[row_ind, 1:])
    images = np.reshape(images, newshape=(-1, 28, 28,1))
    labels = np.array(labels)
    return images, labels

training_images, training_labels = get_data('../input/Kannada-MNIST/train.csv')
val_images, val_labels = get_data('../input/Kannada-MNIST/Dig-MNIST.csv')
test_images, test_id = get_data('../input/Kannada-MNIST/test.csv')

print(training_images.shape)
print(training_labels.shape)
print(test_images.shape)


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


training_labels = to_categorical(training_labels, num_classes = 10)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(training_images,training_labels, test_size = 0.1, random_state=10)


# In[ ]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop


# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.compile(optimizer = RMSprop(lr=0.001) , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.fit(X_train, Y_train, batch_size = 80, epochs =10, validation_data = (X_val, Y_val), verbose = 2)


# In[ ]:


results = model.predict(test_images)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("first.csv",index=False)


# In[ ]:




