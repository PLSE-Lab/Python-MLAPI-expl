#!/usr/bin/env python
# coding: utf-8

# # Sound Keras Hello World
# This kernel attempts to provide a minimal Convolutional Neural Network trained on sound files.
# We use the Keras api for our model definition.
# We use the [British Birdsong dataset](https://www.kaggle.com/rtatman/british-birdsong-dataset) because it has more than 2 categories and is easily available in Kaggle Kernels.
# 
# It will generate a spectrogram from the sound, and will use typical 2d image processing techniques on this representation. This is a common path.
# 
# Future versions may stick to only the 1D signal and convolutions thereof (this is in some sense a purer approach but I'm less experienced with it).

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt


# In[ ]:


# install in kaggle kernel
# if your sound files are .wav, scipy.io.wavfile is more reliable
# this module sometimes prevents us from using kaggle GPU's
import soundfile as sf


# In[ ]:


train_dir = '../input/songs/songs'


# In[ ]:


df = pd.read_csv('../input/birdsong_metadata.csv')
df.head()


# In[ ]:


# find NaNs in metadata
df[df.isnull().any(axis=1)]


# Looks like some samples are missing lat lon values. For our task this is not an issue so we will not be handling the missing data.

# In[ ]:


# num of samples
len(df)


# In[ ]:


df[['genus', 'species']].describe()


# In[ ]:


# load sounds from disk
# takes 1 minute
paths = [os.path.join(train_dir, x) for x in os.listdir(train_dir)]
dataset = []
for p in tqdm(paths):
    audio, _ = sf.read(p)
    dataset.append(audio)


# In[ ]:


dataset[1].shape


# In[ ]:


# TODO zeropad samples to max length, without converting away from numpy array
dataset = keras.preprocessing.sequence.pad_sequences(dataset)
dataset


# In[ ]:


# convert list of numpy arrays into single numpy array for consumption
x_train = np.array(dataset)
x_train.shape


# In[ ]:


# naive label object to get keras model to compile.
# TODO convert np.array of sample labels into one-hot-vector-label
y_train = keras.utils.to_categorical([0, 1, 2], num_classes=3)
y_train


# In[ ]:


# Define model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# Train model
model.fit(x_train, y_train, epochs=10, batch_size=32)


# In[ ]:


# TODO use model to predict category on test set

