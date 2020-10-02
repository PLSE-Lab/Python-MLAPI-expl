#!/usr/bin/env python
# coding: utf-8

# # Supervised learning
# This task is supervised learning. Training data is labeled.  
# 
# ### Third try
# This is my third try. I recently leaned how the neural networks are build and each layer's function for a test. This is an experiment for how those knowledge works.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Note
# 1. We choose the model for image data. Convolutional Neural Networks (CNN) is the model.  What kind of CNN will be working well? 

# In[ ]:


# path
train_path = '../input/train.csv'
test_path = '../input/test.csv'


# In[ ]:


# load data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_shape = train_data.shape
train_shape


# In[ ]:


test_shape = test_data.shape
test_shape


# ## Data

# In[ ]:


X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_finaltest = test_data


# In[ ]:


# show example image
some_digit = np.array(X_train.iloc[0])
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
          interpolation = 'nearest')
plt.axis('off')
plt.show()


# In[ ]:


# check that label
y_train.iloc[0]


# # Model

# In[ ]:


# reshape input : 28*28 = 784 (2D)
X_train = np.array(X_train).reshape(train_shape[0], 784).astype('float32')
X_test = np.array(X_finaltest).reshape(test_shape[0], 784).astype('float32')


# In[ ]:


X_train.shape


# In[ ]:


# normalize input
# max pixel size is 255
X_train /= 255
X_test /= 255

# from sklearn.model_selection import train_test_split
train_1, test_1, label_train, label_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
# # LeNet
# - Try with batch normalization

# In[ ]:


# convert class vectors to binary class matrices
# https://keras.io/utils/
label_train = np_utils.to_categorical(label_train, 10)
label_test = np_utils.to_categorical(label_test, 10)


# In[ ]:


# Keras module
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[ ]:


input_shape = train_1.shape


# In[ ]:


model = Sequential()
model.add(Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
model.add(BatchNormalization(axis=1)
# model.add(Dense(120, input_shape = (784, ), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


model.fit(train_1, label_train, batch_size=20, epochs=15,
         validation_data=(train_1, label_train), verbose=2)


# In[ ]:


# evaluating the model
score = model.evaluate(test_1, label_test, verbose=0)
print("Accuracy: ", score[1])


# In[ ]:


reversed_X_test = X_test * 255
display(reversed_X_test)
display(X_test.shape)


# In[ ]:


# try this model to test.csv
result_label = model.predict_classes(reversed_X_test,batch_size=12, verbose=1)


# In[ ]:


display(result_label)
display(result_label.shape)


# In[ ]:


type(result_label)


# # submission data

# In[ ]:


# make a dataframe for submission data
sub_data = {'ImageID': pd.Series(test_data.index.values + 1),
        'Label': pd.Series(result_label)}

sub_df = pd.DataFrame(data=sub_data)


# In[ ]:


sub_df.to_csv("second_mnist.csv",index=False)


# In[ ]:




