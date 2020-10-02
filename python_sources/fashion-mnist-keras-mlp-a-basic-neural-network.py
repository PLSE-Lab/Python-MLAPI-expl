#!/usr/bin/env python
# coding: utf-8

# **Fashion MNIST - A Multilayer Perceptron**
# 
# Hello Internet! In this kernel, we'll make a simple neural network that gets ~90% accuracy on the Fashion MNIST dataset (a ten class, 28x28 image classification problem).

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

import pandas as pd
import numpy as np


# First up: importing modules. This model just feeds forwards, so we can use a `Sequential` class. As for the layers themselves, we're only using `Dense` and `Activation`. Nothing fancy.

# In[ ]:


#  DEFINE CONSTANTS

INPUT_SHAPE = 784
NUM_CATEGORIES = 10

LABEL_DICT = {
 0: "T-shirt/top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

# LOAD THE RAW DATA
train_raw = pd.read_csv('../input/fashion-mnist_train.csv').values
test_raw = pd.read_csv('../input/fashion-mnist_test.csv').values


# Next some constants. `INPUT_SHAPE` is 784 (28 x 28 - flattened form of the image), and `NUM_CATEGORIES` is 10. All fairly self explanatory.  At the bottom, we use `pd.read_csv` to pull in our data, and we grab the `values` property, which is a numpy array version of the `DataFrame` we just read in.

# In[ ]:


# split into X and Y, after one-hot encoding
train_x, train_y = (train_raw[:,1:], to_categorical(train_raw[:,0], num_classes = NUM_CATEGORIES))
test_x, test_y = (test_raw[:,1:], to_categorical(test_raw[:,0], num_classes = NUM_CATEGORIES))

# normalize the x data
train_x = train_x / 255
test_x = test_x / 255


# Next, we split the import data into training and testing data (as well as X and Y). Any "x" variable is an input, while "y" is the expected output. We set train and test x to everything but the first column of data in our input data (hence the slice), and use Keras' `to_categorical` to one-hot encode the output label to a vector of length `NUM_CATEGORIES` (10). We then normalize the X data. We change the range from 0 - 255 to 0 - 1 by dividing by 255

# In[ ]:


# BUILD THE MODEL
model = Sequential()

model.add(Dense(512, input_dim = INPUT_SHAPE))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(NUM_CATEGORIES))
model.add(Activation('softmax'))

# compile it - categorical crossentropy is for multiple choice classification
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Now for the fun part - defining our model! In this case it's a simple four layer network - an input shape of `INPUT_SHAPE` (784), three 512 neuron layers, and an output layer with `NUM_CATEGORIES` neurons (10). We use categorical crossentroy as our loss, as we've got a multi-class classification problem. For an activation function, we use ReLU all the way, except for the output layer, which uses softmax.

# In[ ]:


# train the model!
model.fit(train_x,
          train_y,
          epochs = 8,
          batch_size = 32,
          validation_data = (test_x, test_y))


# Finally, the training. We tell it to use our `train_x` and `train_y` as our training data, `test_x` and `test_y` to validate, use 32 samples per training pass, and run through the whole dataset 8 times.

# In[ ]:


# how'd the model do?
model.evaluate(train_x, train_y)


# Nice! The first parameter is loss, while the second parameter is accuracy. 90%. Yay!
# 
# **Resources:**
# [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist),
# [Keras](https://keras.io)
