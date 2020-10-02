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

import tensorflow as tf           # tf 2.0


# ### Read data
# Use `test.csv` and `train.csv` provided. Separate training inputs and training targets.

# In[ ]:


test_data = pd.read_csv('../input/test.csv', dtype=np.float64)
train_data = pd.read_csv('../input/train.csv', dtype=np.float64)

train_targets = train_data['label']
train_data = train_data.drop(['label'], axis=1)


# # scale the data
# test_data = test_data / 255.
# train_data = train_data / 255.

# In[ ]:


# scale the data
test_data = test_data / 255.
train_data = train_data / 255.


# ### Get validation data from training data
# 
# In this notebook, I'll use 10% of the training data for validation. 
# 
# **TODO:** shuffle training data and validation data (probably wont make a big difference here, but is good practice.

# In[ ]:


# get validation data from training data
n_validation = int(len(train_data) * 0.1)

validation_inputs = train_data[:n_validation].values
validation_targets = train_targets[:n_validation].values

train_targets = train_targets[n_validation:].values
train_data = train_data[n_validation:].values


# In[ ]:


print(f'validation_inputs: {validation_inputs.shape}')
print(f'validation_targets: {validation_targets.shape}')
print(f'train_data: {train_data.shape}')
print(f'train_targets: {train_targets.shape}')
print(f'test_data: {test_data.shape}')


# ### Defining the model and training the model
# 
# In this model, we define 3 hidden layers using relu activation function (width 500).

# In[ ]:


# Set the input and output sizes
input_size = 784
output_size = 10
hidden_layer_size = 500
    
# define model and layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    
    # activate with softmax since we are outputting a probability
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Choose the optimizer and the loss function
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training

max_epochs = 30

# set an early stopping mechanism, checks if validation loss increases (objective is to prevent overfitting)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
model.fit(train_data,
          train_targets,
          epochs=max_epochs,
          callbacks=[early_stopping],
          validation_data=(validation_inputs, validation_targets),
          verbose = 1
          )


# ### Get results from test data
# 
# Get results from test data and transform output into kaggle submission format:
# 
# ```
# ImageId, Label
# 1, 3
# 2, 8
# 3, 0
# ```

# In[ ]:


predictions = model.predict(test_data.values, verbose=1)


# In[ ]:


imageid = np.arange(1, len(predictions)+1)
labels = [np.argmax(x) for x in predictions]

submission = pd.DataFrame(data={'ImageId': imageid, 'Label': labels})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




