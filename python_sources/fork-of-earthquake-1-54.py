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


# In[ ]:


import numpy as np 
import pandas as pd
import os
from tqdm import tqdm

# Fix seeds
from numpy.random import seed
seed(639)
from tensorflow import set_random_seed
set_random_seed(5944)

# Import
float_data = pd.read_csv("../input/train.csv", nrows=300000, dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values

# Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.
# Can easily be extended. Expects a two dimensional array.
def extract_features(z):
     return np.c_[z.mean(axis=1), 
                  z.min(axis=1),
                  z.max(axis=1),
                  z.std(axis=1)]

# For a given ending position "last_index", we split the last 150'000 values 
# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.
# From each piece, a set features are extracted. This results in a feature matrix 
# of dimension (150 time steps x features).  
def create_X_Y(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and without approximate standardization with mean 5 and std 3.
    chunk = x[(last_index - n_steps * step_length):last_index]
    X = (chunk.T[0].reshape(n_steps, step_length) - 4.5 ) / 10.7
    Y = chunk.T[1].min()
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(X),
                 extract_features(X[:, -step_length // 10:]),
                 extract_features(X[:, -step_length // 100:])],Y
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
       
    assert last_index - n_steps * step_length >= 0

    # Reshaping and without approximate standardization with mean 5 and std 3.
    chunk = x[(last_index - n_steps * step_length):last_index]
    X = (chunk.reshape(n_steps, step_length) - 4.5 ) / 10.7
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[extract_features(X),
                 extract_features(X[:, -step_length // 10:]),
                 extract_features(X[:, -step_length // 100:])]


# In[ ]:


# Query "create_X" to figure out the number of features
features,targets = create_X_Y(float_data[0:150000])
n_features = features.shape[1]
print("Our RNN is based on %i features"% n_features)


# create_X_Y returns X,Y: <br>
# X 150 * 12 matrix<br>
# Y 150 * 1 matrix

# In[ ]:


# define create_dataset
# train = create_dataset ...
# validation = create_dataset ...
# test = create_dataset ...


# In[ ]:


n_train = 5000
float_data = pd.read_csv("../input/train.csv", nrows=n_train*150000, dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
#float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})


# In[ ]:


# len(float_data)/150000
# print(float_data[:5])
a = float_data[:,0]
print(np.mean(a))
print(np.std(a))
len(float_data)


# In[ ]:


# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1
     
    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)
         
        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j],targets[j] = create_X_Y(data, last_index=row, n_steps=n_steps, step_length=step_length)
        yield samples, targets

batch_size = 64

# Position of second (of 16) earthquake. Used to have a clean split
# between train and validation
second_earthquake = 50085877
#float_data[second_earthquake, 1]

tv_bd = 3800 * 150000

# steps = 300
# step_length = 150000/steps
# Initialize generators
train_gen = generator(float_data, max_index = 4000 * 150000, batch_size=batch_size,n_steps=300,step_length=500) # Use this for better score
# train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)
valid_gen = generator(float_data, min_index= 3800 * 150000,batch_size=batch_size,n_steps=300, step_length=500)


# In[ ]:


# Define model
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU,CuDNNLSTM,Bidirectional,Activation,Dropout
from keras.optimizers import adam,RMSprop
from keras.callbacks import ModelCheckpoint

cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]

model = Sequential()
model.add(CuDNNLSTM(64,return_sequences=True,input_shape=(None, n_features)))
model.add(CuDNNGRU(48))
model.add(Dense(16,activation="relu"))
model.add(Dense(1))
#model.add(Dense(16,activation="tanh",kernel_regularizer=regularizers.l2(0.05)))
#model.add(CuDNNGRU(48, input_shape=(None, n_features)))


model.summary()

# Compile and fit model
model.compile(optimizer=RMSprop(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,
                              epochs=100,
                              verbose=2,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=200)


# In[ ]:


# Visualize accuracies
import matplotlib.pyplot as plt

def perf_plot(history, what = 'loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1
    
    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None

perf_plot(history)

# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(tqdm(submission.index)):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x,n_steps=300, step_length=500), 0))

submission.head()

# Save
submission.to_csv('submission.csv')

