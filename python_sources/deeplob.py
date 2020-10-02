#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras import layers
from keras.models import Model
from keras import Input
from keras.layers import LSTM
from keras.utils import to_categorical
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data_file = '/kaggle/input/fi2010/FI2010/FI2010_train.csv'
test_data_file = '/kaggle/input/fi2010/FI2010/FI2010_test.csv'


# In[ ]:


train_data = pd.read_csv(train_data_file, index_col=0)


# In[ ]:


# ready data for training:
# 1. sample_size=100: the most 100 recent updates
# 2. feature_num=40: 40 features per time stamp
# 3. target_num=5: relative changes for the next 1,2,3,5 and 10 events(5 in total)
def data_generator(data, batch_size=32, lookback=100,
                   feature_num=40, target_delay=1, shuffle=False):
    data = data.values
    shape = data.shape
    max_index = shape[0]
    min_index = 0
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index + lookack, max_index)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback, feature_num))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            samples[j] = data[row - lookback: row, 0: feature_num]  # take the first 40 columns as features
            targets[j] = data[row - 1, target_delay - 6]
        samples = samples.reshape(samples.shape[0], samples.shape[1],
                                  samples.shape[2], 1)# add the 4th dimension: 1 channel
        # "Benchmark dataset for mid-price forecasting of limit order book data with machine learning"
        # labels 1: equal to or greater than 0.002
        # labels 2: -0.00199 to 0.00199
        # labels 3: smaller or equal to -0.002
        # Y=Y-1 relabels as 0,1,2
        targets = targets - 1
        targets = targets.astype(int)
        targets = to_categorical(targets, num_classes=3)# y is the next event's mid price (k=1)
        yield samples, targets


# In[ ]:


# the size of a single input is (100,40)
input_tensor = Input(shape=(100,40,1))

# convolutional filter is (1,2) with stride of (1,2)
layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(input_tensor)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (1,10))(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

# Inception Module
tower_1 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
tower_1 = layers.Conv2D(32, (3,1), padding='same')(tower_1)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

tower_2 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
tower_2 = layers.Conv2D(32, (5,1), padding='same')(tower_2)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
tower_3 = layers.Conv2D(32, (1,1), padding='same')(tower_3)
tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

# concatenate features of tower_1, tower_2, tower_3
layer_x = layers.Reshape((100,96))(layer_x)

# 64 LSTM units
layer_x = LSTM(64)(layer_x)
# The last output layer uses a softmax activation function
output = layers.Dense(3, activation='softmax')(layer_x)
model = Model(input_tensor, output)

model.summary()


# In[ ]:


batch_size = 32
steps_per_epoch =  len(train_data) // batch_size
train_gen = data_generator(train_data, batch_size=batch_size, lookback=100,
                   feature_num=40, target_delay=1, shuffle=False)
opt = keras.optimizers.Adam(lr=0.01, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit_generator(train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100)

