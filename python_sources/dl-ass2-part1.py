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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf

import numpy as np

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adagrad, SGD, RMSprop, Adam
from tensorflow.train import AdamOptimizer


# # Load and prepare data

# Letters Datset - Separate train into train and validation (and stratify)

# In[ ]:


train_features_letters = np.load('../input/transferlearning/imagesLettersTrain.npy').astype('float32') 
train_labels_letters = np.load('../input/transferlearning/labelsTrain.npy')

train_features_letters, validation_features_letters, train_labels_letters, validation_labels_letters =     train_test_split( train_features_letters, train_labels_letters, test_size=0.30, random_state=42,stratify=train_labels_letters)

test_features_letters = np.load('../input/transferlearning/imagesLettersTest.npy').astype('float32')
test_labels_letters = np.load('../input/transferlearning/labelsTest.npy')

train_labels_letters = tf.keras.utils.to_categorical(train_labels_letters, 26)
validation_labels_letters = tf.keras.utils.to_categorical(validation_labels_letters, 26)
test_labels_letters = tf.keras.utils.to_categorical(test_labels_letters, 26)

train_features_letters = train_features_letters.reshape((train_features_letters.shape[0], 28, 28,1)) / 255.0
validation_features_letters = validation_features_letters.reshape((validation_features_letters.shape[0], 28, 28,1)) / 255.0
test_features_letters = test_features_letters.reshape((test_features_letters.shape[0], 28, 28, 1)) / 255.0


# MNIST dataset - Separate train into train and validation (and stratify)*

# In[ ]:


x_train = pd.read_csv('../input/mnist-dataset/mnist_x_train.csv').values[:,1:]
y_train = pd.read_csv('../input/mnist-dataset/mnist_y_train.csv').values[:,1:]
x_test = pd.read_csv('../input/mnist-dataset/mnist_x_test.csv').values[:,1:]
y_test = pd.read_csv('../input/mnist-dataset/mnist_y_test.csv').values[:,1:]

x_train, x_validation, y_train, y_validation =     train_test_split( x_train, y_train, test_size=0.30, random_state=42,stratify=y_train)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_validation = x_validation.reshape((x_validation.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = x_train.astype('float32') / 255.0
x_validation = x_validation.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_validation = tf.keras.utils.to_categorical(y_validation, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# # Build model architecture for MNIST dataset

# In[ ]:


Inputs = Input(shape = (28, 28, 1), name = 'Inputs')

Layer = Conv2D(128, (3,3), padding = 'same', input_shape = (28, 28, 1))(Inputs)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(64, (3,3), padding = 'same')(Layer)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(32, (3,3), padding = 'same')(Layer)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)

Layer_x = Flatten(name='Layer_x')(Layer)

Layer = Dense(10, activation = 'softmax')(Layer_x)


# In[ ]:


mnist_model = Model(inputs = Inputs, outputs = Layer)
mnist_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'] )
mnist_model.summary()


# ## Train MNIST

# In[ ]:


epochs = 35
batch_size = 50

mnist_model.fit(x_train, y_train, validation_data = (x_validation, y_validation), batch_size = batch_size, epochs = epochs)


# # Test MNIST

# In[ ]:


def get_label(arr):
    return np.where(arr == 1)[0]


# In[ ]:


pred = mnist_model.predict(x_test)
# convert one-hot to true value
true_test = np.apply_along_axis(get_label,1,y_test).flatten()
# convert softmax to true value
pred = np.apply_along_axis(np.argmax,1,pred)
# aclculate accuracy
acc = (pred == true_test).sum() / pred.shape[0]
print(f"Accuracy of the model on the test set: {acc}")


# In[ ]:


# finally save the weights of the model
mnist_model.save_weights("saved_models/mnist_weights")


# # Letters dataset

# Now we want to transfer the weights of the convolution layers from the mnist neural network to this new network. <br>
# For that we need to have the exact same structure of the convolution part, so let's replicate the convolution part of the mnist network!

# In[ ]:


Inputs = Input(shape = (28, 28, 1), name = 'Inputs')

Layer = Conv2D(128, (3,3), padding = 'same', input_shape = (28, 28, 1))(Inputs)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(64, (3,3), padding = 'same')(Layer)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(32, (3,3), padding = 'same')(Layer)
Layer = Activation('relu')(Layer)
Layer = BatchNormalization()(Layer)
Layer_x = Flatten(name='Layer_x')(Layer)


# In[ ]:


old_model = Model(inputs = Inputs, outputs = Layer_x)
old_model.load_weights('saved_models/mnist_weights')

# We don't want to train the convolutional layers
# because they are already trained
for each_layer in old_model.layers:
    each_layer.trainable = False

old_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'] )
old_model.summary()


# In[ ]:


# What's different between this architecture and the 
# last one is the last layer which this time will 
# have 26 neurons, one for each letter in the alphabet.
layer = Dense(26, activation='softmax')(old_model.get_layer('Layer_x').output)
# And now define the letter's model using 
# the mnist one plus the layer we just created.
model_letters = Model(inputs = old_model.get_layer('Inputs').output, outputs = layer)
model_letters.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
model_letters.summary()


# In[ ]:


for i,layer in enumerate(model_letters.layers):
    print(i, layer.name, layer.trainable)


# # Train Letters

# In[ ]:


# And let's now train!
n_epochs = 100
batch = 30

model_letters.fit(train_features_letters, train_labels_letters,validation_data = (validation_features_letters, validation_labels_letters), batch_size = batch, epochs = n_epochs)


# # test Labels

# In[ ]:


pred = model_letters.predict(test_features_letters)
# convert one-hot to true value
true_test = np.apply_along_axis(get_label,1,test_labels_letters).flatten()
# convert softmax to true value
pred = np.apply_along_axis(np.argmax,1,pred)
# aclculate accuracy
acc = (pred == true_test).sum() / pred.shape[0]
print(f"Accuracy of the model on the test set: {acc}")


# # Now let's train the same architecture for the Letters dataset, but without loading the weights from the MNIST dataset

# In[ ]:


Inputs = Input(shape = (28, 28, 1), name = 'Inputs')

Layer = Conv2D(128, (3,3), padding = 'same', input_shape = (28, 28, 1))(Inputs)
Layer = BatchNormalization()(Layer)
Layer = Activation('relu')(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(64, (3,3), padding = 'same')(Layer)
Layer = BatchNormalization()(Layer)
Layer = Activation('relu')(Layer)
Layer = MaxPool2D(pool_size = (2,2))(Layer)
Layer = Conv2D(32, (3,3), padding = 'same')(Layer)
Layer = BatchNormalization()(Layer)
Layer = Activation('relu')(Layer)

Layer_x = Flatten(name='Layer_x')(Layer)

Layer = Dense(26, activation = 'softmax')(Layer_x)
# And now define the letter's model using 
# the mnist one plus the layer we just created.
model_letters_empty = Model(inputs = Inputs, outputs = Layer)
model_letters_empty.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
model_letters_empty.summary()


# In[ ]:


# And let's now train!
n_epochs = 10
batch = 30

model_letters_empty.fit(train_features_letters, train_labels_letters,validation_data = (validation_features_letters, validation_labels_letters), batch_size = batch, epochs = n_epochs)


# In[ ]:


pred = model_letters_empty.predict(test_features_letters)
# convert one-hot to true value
true_test = np.apply_along_axis(get_label,1,test_labels_letters).flatten()
# convert softmax to true value
pred = np.apply_along_axis(np.argmax,1,pred)
# aclculate accuracy
acc = (pred == true_test).sum() / pred.shape[0]
print(f"Accuracy of the model on the test set: {acc}")

