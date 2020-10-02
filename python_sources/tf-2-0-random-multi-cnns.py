#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


# Import packages

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, utils


# In[ ]:


tf.__version__


# # Data Preprocessing

# In[ ]:


# Load the data
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
submit_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

# Train data
train_df.head()


# In[ ]:


X = train_df.iloc[:,1:].values.reshape(-1, 28, 28, 1)/255.0
Y = train_df.label.values
Y = utils.to_categorical(Y)
X_test = test_df.values.reshape(-1, 28, 28, 1)/255.0


# In[ ]:


def split_data(seed):
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,
                                                          test_size=0.1,
                                                          random_state=seed)
    return X_train, X_valid, Y_train, Y_valid


# In[ ]:


num_classes = 10
shape = (28, 28, 1)

min_hidden_layers = 3
max_hidden_layers = 6
min_filters = 32
max_filters = 128
dropout = 0.4

num_models = 15
num_epochs = 30


# In[ ]:


random_CNNs = [0] * num_models

for i in range(num_models):
    
    # Build Model
    Filters = list(range(min_filters, max_filters))
    Layers = list(range(min_hidden_layers, max_hidden_layers))
    Layer = random.choice(Layers)
    Filter = random.choice(Filters)

    random_CNNs[i] = models.Sequential()
    random_CNNs[i].add(layers.Conv2D(Filter, (3, 3), padding='same', input_shape=shape))
    random_CNNs[i].add(layers.Activation('relu'))
    random_CNNs[i].add(layers.Conv2D(Filter, (3, 3)))
    random_CNNs[i].add(layers.Activation('relu'))

    for j in range(0, Layer):
        Filter = random.choice(Layers)
        random_CNNs[i].add(layers.Conv2D(Filter, (2, 2), padding='same'))
        random_CNNs[i].add(layers.Activation('relu'))
        random_CNNs[i].add(layers.Dropout(dropout))

    random_CNNs[i].add(layers.Flatten())
    random_CNNs[i].add(layers.Dense(128, activation='relu'))
    random_CNNs[i].add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile
    random_CNNs[i].compile(optimizer='adam',
                                 loss='binary_crossentropy',
                                 metrics=['accuracy'])
    
    # Split Data
    X_train, X_valid, Y_train, Y_valid = split_data(i+137)
    
    # Train Model
    print('TRAINING MODEL %d:' % (i+1))
    random_CNNs[i].fit(X_train, Y_train, epochs=num_epochs,
                             validation_data=(X_valid, Y_valid))
    
    # Predict
    pred = random_CNNs[i].predict(X_test)
    pred = np.argmax(pred,axis = 1)
    
    model_name = 'model_' + str(i)
    ensemble_df = submit_df.copy()
    ensemble_df[model_name] = pred
    ensemble_df.head()


# In[ ]:


# Final prediction
final_pred = ensemble_df.iloc[:,2:].mode(axis=1).iloc[:,0]
submit_df.Label = final_pred.astype(int)
submit_df.head()


# In[ ]:


# Create a submission file
submit_df.to_csv('submission.csv', index=False)


# In[ ]:




