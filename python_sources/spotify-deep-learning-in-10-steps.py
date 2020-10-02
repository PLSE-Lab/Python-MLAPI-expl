#!/usr/bin/env python
# coding: utf-8

# ### Context
# * This is a dataset consisting of features for tracks fetched using Spotify's Web API. 
# * The tracks are labeled '1' or '0' ('Hit' or 'Flop') depending on some criteria of the author.
# * This dataset can be used to make a classification model that predicts whether a track would be a 'Hit' or not.
# <br>
# <br>
# <font color = 'blue'>
# 1. Content
#     * [Load Libraries](#1)
#     * [Load Dataset](#2)    
#     * [Balance The Dataset](#3)
#     * [Shuffle The Data](#4)
#     * [Standardize The Inputs](#5)
#     * [Split The Dataset into Train,Validation and Test](#6)
#     * [Create The Deep Learning Algorithm](#7)
#     * [Visualize Neural Network Loss History](#8)
#     * [Test The Model](#9)

# <a id = "1"></a><br>
# ## Load Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(0)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <a id = "2"></a><br>
# ## Load Data
# * I will use just 70s that has much more datas.

# In[ ]:


data = pd.read_csv("/kaggle/input/the-spotify-hit-predictor-dataset/dataset-of-70s.csv")
data.head()


# <a id = "3"></a><br>
# ## Balance The Dataset
# * This is a balanced dataset

# In[ ]:


data.target.value_counts()


# <a id = "4"></a><br>
# ## Shuffle The Data

# In[ ]:


data = data.sample(frac=1)
data.head()


# <a id = "5"></a><br>
# ## Standardize The Inputs

# In[ ]:


data.info()


# ### Drop Categorical Features

# In[ ]:


data.drop(["track","artist","uri"],axis=1,inplace=True)


# In[ ]:


unscaled_inputs = data.iloc[:,0:-1]
target = data.iloc[:,[-1]]


# In[ ]:


scaled_inputs = preprocessing.scale(unscaled_inputs)


# <a id = "6"></a><br>
# ## Split The Dataset into Train,Validation and Test
# * 80% , 10% , 10%

# In[ ]:


samples_count = scaled_inputs.shape[0]
#
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count


# In[ ]:


# train:
train_inputs = scaled_inputs[:train_samples_count]
train_targets = target[:train_samples_count]


# In[ ]:


# validation:
validation_inputs = scaled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = target[train_samples_count:train_samples_count+validation_samples_count]


# In[ ]:


# test:
test_inputs = scaled_inputs[train_samples_count+validation_samples_count:]
test_targets = target[train_samples_count+validation_samples_count:]


# In[ ]:


# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)


# ## Save the three datasets in *.npz

# In[ ]:


# Save the three datasets in *.npz.
# We will see that it is extremely valuable to name them in such a coherent way!

np.savez('Spotify_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Spotify_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Spotify_data_test', inputs=test_inputs, targets=test_targets)


# <a id = "7"></a><br>
# ## Create The Deep Learning Algorithm

# ### Data

# In[ ]:


npz = np.load('Spotify_data_train.npz')
# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('Spotify_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('Spotify_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# ### Model

# In[ ]:


# Set the input and output sizes
input_size = 15 # count of features
output_size = 2 # count of targets
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 50 # counts of neurons
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


# ### Choose the optimizer and the loss function

# In[ ]:


# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
#custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# ### Training

# In[ ]:


# That's where we train the model we have built.
# set the batch size
batch_size = 300
# set a maximum number of training epochs
max_epochs = 6

# fit the model
# note that this time the train, validation and test data are not iterable
history = model.fit(  train_inputs, # train inputs
                      train_targets, # train targets
                      batch_size=batch_size, # batch size
                      epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
                      # callbacks are functions called by a task when a task is completed
                      # task here is to check if val_loss is increasing
                      #callbacks=[early_stopping], # early stopping
                      validation_data=(validation_inputs, validation_targets), # validation data
                      verbose = 2 # making sure we get enough information about the training process
          )  


# <a id = "8"></a><br>
# ## Visualize Neural Network Loss History

# In[ ]:


# Get training and test loss histories
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


# <a id = "9"></a><br>
# ## Test The Model

# In[ ]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

