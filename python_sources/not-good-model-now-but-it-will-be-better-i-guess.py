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


get_ipython().system('pip install tensorflow==2.0.0-alpha0')


# In[ ]:


import os
import sys
import gc
import psutil

from os.path import isdir, join
from time import time
from pathlib import Path
import pandas as pd

import numpy as np
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA

from IPython.display import display

print('Finish Import Utilities libary')


# In[ ]:


from time import time
import gc
import tensorflow as tf
from random import randint
from tensorflow import keras
from tensorflow.keras import activations, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Input, Dropout, Flatten
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau
print('Finish import model library')


# In[ ]:


def get_data(path):
    '''Get data from the path and create a pandas dataframe to store it,
    the function will return a pandas dataframe with fpath(file path) and label'''
    
    label_list = []
    fname = []
    valid_label = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

    # Get every audio files path and label
    files = [(str(file), file.parts[-2]) for file in Path(path).glob("**/*.wav") if file]
    file_len = len(files)
    print('Finish getting data')
    
    # Valid label name
    for file in files:
        if file[1] == '_background_noise_':
            label = 'silence'
        elif file[1] not in valid_label:
            label = 'unknown'
        else:
            label = file[1]
            
        # Normal version training set, but only got 0.6 points in kaggle
        label_list.append(label)
        fname.append(file[0])

        # Try to only train valid-label data
#         if label in valid_label:
#             label_list.append(label)
#             fname.append(file[0])
        
    data = pd.DataFrame({'fpath': fname, 'label': label_list})
    
    print('Finish appending array')
    return data


# In[ ]:


# Get train data set
train_df = get_data('./train/audio')

# Seperate train set and validation set
train_set = train_df.sample(frac=0.8, replace=False, random_state=42)
valid_set = train_df.loc[set(train_df.index) - set(train_set.index)]

y_train = np.array(train_set.label)
y_train = pd.get_dummies(y_train, dtype=bool)
x_train = np.array(train_set.fpath)

y_valid = np.array(valid_set.label)
y_valid = pd.get_dummies(y_valid, dtype=bool)
x_valid = np.array(valid_set.fpath)

display(train_set.info())
display(valid_set.info())
gc.collect()

print(len(x_valid), len(y_valid))


# In[ ]:


def get_model(shape):
    '''Create a keras functional model'''
    
    inputlayer = Input(shape=shape)
    
    # Nornal model
    nclass = 12
    
    # Experience model
#     nclass = 11
    
    norm_input = BatchNormalization()(inputlayer)
    model = Conv2D(16, kernel_size=2, padding='same', activation=activations.relu)(norm_input)
    model = Conv2D(16, kernel_size=2, padding='same', activation=activations.relu)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(rate=0.2)(model)
    model = Conv2D(32, kernel_size=3, padding='same', activation=activations.relu)(model)
    model = Conv2D(32, kernel_size=3, padding='same', activation=activations.relu)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(rate=0.2)(model)
    model = Conv2D(64, kernel_size=3, padding='same', activation=activations.relu)(model)
    model = MaxPool2D(pool_size=(2, 2))(model)
    model = Dropout(rate=0.2)(model)
    model = Flatten()(model) 

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(model))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inputlayer, outputs=dense_1)
    model.compile(optimizer='adam', loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[ ]:


shape = (99, 161, 1)
model = get_model(shape)
# model.summary()


# In[ ]:


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def get_spectrogram(paths, y=None, nsamples=16000):
    wavs = [wavfile.read(path)[1] for path in paths]
    
    data = []
    label = []
    for wav in wavs:
        try:
            if wav.size < 16000:
                d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
            else:
                d = wav[0:nsamples]
            data.append(d)
        except:
            pass

    spg = [log_specgram(d, nsamples)[2] for d in data]
    spg = [s.reshape(99, 161, -1) for s in spg]
    return (spg)

# Need to fix if wavfile.read() fail, the length of [spgs] would not be the same as y
def batch_generator(x, y, batch_size=16):
    # Return a random image from X, y
    ylen = len(y)
    loopcount = ylen // batch_size
    while True:
        i = randint(0,loopcount)
        x_list = x[i * batch_size:(i + 1) * batch_size]
        spgs = get_spectrogram(x_list)
        
        yield np.concatenate([spgs]), y[i * batch_size:(i + 1) * batch_size]


# In[ ]:


# Train the model
batch_size = 10
epochs = 12
path = './tensorboard/keras_' + str(time())
history = model.fit_generator(
    generator=batch_generator(x_train, y_train, batch_size),
    validation_data=batch_generator(x_valid, y_valid, batch_size),
    epochs=epochs,
    steps_per_epoch=y_train.shape[0] // batch_size,
    validation_steps=y_valid.shape[0] // batch_size,
#     callbacks=[TensorBoard(log_dir=path)],
    verbose=1,
)

# Save the model
# path = './model/model_' + str(time()) + '.h5'
# model.save(path)


# In[ ]:


'''log_specgram() and get_spectrogram() defined before '''

def get_prediction(path, model, pred_list, nsamples=16000):
    '''Predict the test files and return a pandas dataframe with submission format'''
    prediction = []
    file_name = []
    
    # Get every files path and file's label in test directory
    file_names = [(str(file), file.parts[-2]) for file in Path(path).glob("**/*.wav") if file]
    i, file_len = 0, len(file_names)
    
    start_time = time()
    for name in file_names[0:None]:
        try:
            spg = get_spectrogram([name[0]])
            pred = model.predict(np.array(spg))

            # Add threshold to prediction
            if (pred.max() > 0.5):    
                pred = np.argmax(pred, axis=1)
                pred = pred_list[pred[0]]
            else:
                pred = 'unknown'
        except:
            pred = 'unknown'
        prediction.append(pred)
        file_name.append(name[0].split('/')[-1])

        # Fancy progress bar
        i = i + 1
        if i % 100 == 0:
            print("%d/%d, time: %.4f s" % (i, file_len, time() - start_time), end='\r')
    print('\nFinish prediction')

    submission = pd.DataFrame({'fname': file_name, 'label': prediction})
    return submission
submission = get_prediction('./test/audio/', model1, pred_list)
submission.to_csv('submission1.csv', index=False)
print('Saved csv file')
print(pd.value_counts(submission.label.values))

