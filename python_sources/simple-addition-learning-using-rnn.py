#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# # Making the Model learn Addition!!
# Given the string "54+7", the model should return a prediction: "61".
# 
# Here is my Github repo on [this](https://github.com/digs1998/RNN-implementations) Do refer this and fork my repository

# # Workflow 
# 1. Loading the Libraries
# 2. Generating the Dataset
# 3. Create the Model
# 4. Vectorize and De-Vectorize the Data
# 5. Training the model and Predictions 

# # Basic RNN Understanding
# 
# A **recurrent neural network (RNN)** is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs.This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition
# 
# ![rnn](https://miro.medium.com/max/1254/1*go8PHsPNbbV6qRiwpUQ5BQ.png)
# 
# More Detailed Info can be found on [Medium](https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e)

# # Loading the Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, SimpleRNN, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

from termcolor import colored

print('Tested with tensorflow version 2.0.1')
print('Using tensorflow version:', tf.__version__)


# # Generate Dataset
# Let's start with assigning a variable to all the numbers from 0 to 9 in string format, + indicates to add

# In[ ]:


all_chars = '0123456789+'


# In[ ]:


num_features = len(all_chars)

char_to_index = dict((c, i) for i, c in enumerate(all_chars))
index_to_char = dict((i, c) for i, c in enumerate(all_chars))

print('Number of features:', num_features)


# In[ ]:


def generate_data():
    first_num = np.random.randint(low=0,high=100)
    second_num = np.random.randint(low=0,high=100)
    example = str(first_num) + '+' + str(second_num)
    label = str(first_num+second_num)
    return example, label

generate_data()


# **So we have generated the first data of what we would be using for this model development**

# # Create a Model
# Here we specify the input parameters for an RNN model

# In[ ]:


hidden_units = 128
max_time_steps = 5

model = Sequential([
    SimpleRNN(hidden_units, input_shape=(None, num_features)),
    RepeatVector(max_time_steps),
    SimpleRNN(hidden_units, return_sequences=True),
    TimeDistributed(Dense(num_features, activation='softmax'))
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# # Vectorize and De-Vectorize data
# We convert these datasets into vectorized and de-vectorized form for the model to work efficiently as Machine Learning models require numerical values to compute

# In[ ]:


def vectorize_example(example, label):
    
    x = np.zeros((max_time_steps, num_features))
    y = np.zeros((max_time_steps, num_features))
    
    diff_x = max_time_steps - len(example)
    diff_y = max_time_steps - len(label)
    
    for i, c in enumerate(example):
        x[diff_x+i, char_to_index[c]] = 1
    for i in range(diff_x):
        x[i, char_to_index['0']] = 1
    for i, c in enumerate(label):
        y[diff_y+i, char_to_index[c]] = 1
    for i in range(diff_y):
        y[i, char_to_index['0']] = 1
        
    return x, y

e, l = generate_data()
print('Text Example and Label:', e, l)
x, y = vectorize_example(e, l)
print('Vectorized Example and Label Shapes:', x.shape, y.shape)


# In[ ]:


def devectorize_example(example):
    result = [index_to_char[np.argmax(vec)] for i, vec in enumerate(example)]
    return ''.join(result)

devectorize_example(x)


# # Create Dataset

# In[ ]:


def create_dataset(num_examples=2000):

    x_train = np.zeros((num_examples, max_time_steps, num_features))
    y_train = np.zeros((num_examples, max_time_steps, num_features))

    for i in range(num_examples):
        e, l = generate_data()
        x, y = vectorize_example(e, l)
        x_train[i] = x
        y_train[i] = y
    
    return x_train, y_train

x_train, y_train = create_dataset()
print(x_train.shape, y_train.shape)


# In[ ]:


devectorize_example(x_train[0])


# In[ ]:


devectorize_example(y_train[0])


# ***So we are able to get the representation of data as shown above now our dataset is ready for model based evaluations!!***

# # Training and Predictions

# In[ ]:


simple_logger = LambdaCallback(
    on_epoch_end=lambda e, l: print('{:.2f}'.format(l['val_accuracy']), end=' _ ')
)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(x_train, y_train, epochs=500, validation_split=0.2, verbose=False,
         callbacks=[simple_logger, early_stopping])


# In[ ]:


x_test, y_test = create_dataset(num_examples=20)
preds = model.predict(x_test)
full_seq_acc = 0

for i, pred in enumerate(preds):
    pred_str = devectorize_example(pred)
    y_test_str = devectorize_example(y_test[i])
    x_test_str = devectorize_example(x_test[i])
    col = 'green' if pred_str == y_test_str else 'red'
    full_seq_acc += 1/len(preds) * int(pred_str == y_test_str)
    outstring = 'Input: {}, Out: {}, Pred: {}'.format(x_test_str, y_test_str, pred_str)
    print(colored(outstring, col))
print('\nFull sequence accuracy: {:.3f} %'.format(100 * full_seq_acc))


# In[ ]:




