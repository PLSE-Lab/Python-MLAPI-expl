#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import keras
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)


# In[ ]:


image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')


# ## **MNIST Dataset**
# 
# Modified National Institute of Standards and Technology database
# 
# It is a large dataset of handwritten digits used for training image processing systems
# 
# Training data was taken from American Census Bureau employees 
# Testing data was taken from American high school students
# 
# **Features**
# 1. 60,000 training images , 10,000 testing images
# 2. Extended version of MNIST is called EMNIST published in 2017
# 3. Each image is 28x28 and linearized vector of 1x784
# 
# ![mnist dataset](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-MNIST-Dataset.png)

# **Processing the data**
# 
# *   The MNIST dataset of digits are with structure (nb_samples, 28, 28) with 2D per image of 28x28 pixel greyscale.
# 
# *   But the *Convolution2D* layers in Keras are designed to work on 3D per example.
# 
# *   There are 4D (nb_samples, nb_channels, width, height) with deep layers and each example should become set of feature maps.
# 
# *   Thus the examples of MNIST requires different CNN layer design or a parameter , constructor layer to accept this shape.
# 
# *   *Thus matrix reshape is done to accept this change (60000, 28, 28, 1) - Padding*
# 

# In[ ]:


# x_train.shape[0] = 60000 and x_test.shape[0] = 10000

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# **Converting class vectors to Binary class matrices - One hot encoding**
# 
# Our model cannot work on categorical data directly thus **one hot encoding** is performed.
# 
# The digits from 0 - 9 are represented as set of 9 - 0's and 1 - 1's. Where based on the position of 1 the digit is identified.
# 
# eg) 3 as [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# In[ ]:


y_train = np_utils.to_categorical(y_train, num_classes=None)

y_test = np_utils.to_categorical(y_test, num_classes=None)


# In[ ]:


# Modifying the value of each pixel in range of 0 to 1 - improves the learning rate of model

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Grey scale values and pixel value ranges from 0 - 255
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# **Create the model**
# 
# CNN model with convolutional and pooling layers. Works better for images with grid structures, CNN works well for image classification problems.
# 
# Our convolutional layers will have `64 neurons`(feature maps) and `3x3 feature detector`
# 
# In `max pooling 2x2` matrix. In Keras, a Dense layer implements the operation `output = activation(dot(input, weights) + bias)`
# 
# **Convolutional Layers**
# 
# ![alt text](https://miro.medium.com/max/1332/1*V7YGj0ZWil9V-i0k74QVSQ.png)
# 
# **Pooling Layer**
# 
# ![alt text](https://miro.medium.com/max/608/1*oVOUhBIi59Gb5w7eBzqYuA.png)

# **Parameters**
# 
# `num_classes` - Number of classifier outputs
# 
# `batch_size` - Number of samples propagated through network, from total trainig samples the algorithm takes first 128 training dataset from (1 to 128) and trains the network. And subsequently trains with set of 128 till end.
# 
# `epochs` - Number of times the training vectors are used to update weights
# 
# `Dropout` - Deactivate some neurons while training, to reduce over fitting of model
# 
# **Activation function**
# 
# `relu` - Commonly used for hidden layers. `max(y,0)` where y is the summation of (Weights and Inputs) + Bias

# In[ ]:


batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# **Training the Model**
# 
# `model.fit()` - Start the training of the model, takes training data, validation data , epochs and batch size
# 
# After the training of the model, saving the model and definition in 'mnist.h5' file

# In[ ]:


# Trains the model for fixed number of epochs (iterations on the dataset)
my_model = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

# .h5 is Hierarchical Data Format (HDF). Contains multidimensional arrays of data
model.save('mnist.h5')
print("Model saved as mnist.h5")


# **Evaluating Our Model**
# 
# In test data set we have around 10,000 images. The testing data was not involved in training data.

# In[ ]:


score = model.evaluate(x_test, y_test, verbose = 1)

# To reduce the loss we have used Adadelta Optimizer in training our model
print("Test Loss" , score[0])
print("Test Accuracy" , score[1])


# In[ ]:


image_index = 3333 #Data at index 3333 is 7
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())


# Digit Recognition using CNN in MNIST dataset. Basic intro towards Keras, CNN, optimizers with detailed explanation.
# 
# ## *If you like it, Please upvote!*
