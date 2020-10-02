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


# Our own imports
import json
from glob import glob
from tensorflow import keras
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from keras import optimizers
from keras.models import Model
import pydicom
import pandas as pd
from collections import defaultdict
import keras
import numpy as np


# In[ ]:


# via https://www.kaggle.com/seesee/full-dataset
train_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-train/*/*/*.dcm'))
test_fns = sorted(glob('../input/siim-train-test/siim/dicom-images-test/*/*/*.dcm'))
rles = pd.read_csv('../input/siim-train-test/siim/train-rle.csv')

rles_ = defaultdict(list)
for image_id, rle in zip(rles['ImageId'], rles[' EncodedPixels']):
    rles_[image_id].append(rle)
rles = rles_
annotated = {k: v for k, v in rles.items() if v[0] != ' -1'}


# In[ ]:


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# In[ ]:


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(1024,1024), n_channels=1,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.expand_dims(pydicom.read_file(ID).pixel_array, axis=2)
            
            stripped_id = ID.split('/')[-1][:-4]
            rle = self.labels.get(stripped_id)
            
            if rle is None:
                y[i,] = np.zeros((1024, 1024, 1))
            else:
                if len(rle) == 1:
                    y[i,] = np.expand_dims(rle2mask(rle[0], self.dim[0], self.dim[1]).T, axis=2)
                else: 
                    y[i,] = np.zeros((1024, 1024, 1))
                    for x in rle:
                        y[i,] =  y[i,] + np.expand_dims(rle2mask(x, 1024, 1024).T, axis=2)

        return X, y


# In[ ]:


# 2D blocks
def conv2D_block(inputs, filters, activation, padding, batchnorm=False):
    conv = Conv2D(filters, 3, activation=activation, padding=padding)(inputs)
    if batchnorm:
        conv = BatchNormalization()(conv)
    conv = Conv2D(filters, 3, activation=activation, padding=padding)(conv) 
    if batchnorm:
        conv = BatchNormalization()(conv)
    return conv

def conv2D_maxpool_block(inputs, filters, activation, padding, batchnorm=False):
    conv = conv2D_block(inputs, filters, activation, padding)
    pool = MaxPooling2D()(conv)
    return pool, conv

def upsamp_conv2D_block(conv_prev, conv_direct, filters, activation, padding, batchnorm=False):
    up = UpSampling2D()(conv_prev)
    conc = concatenate([up, conv_direct])
    cm = conv2D_block(conc, filters, activation, padding, batchnorm)
    return cm

def build_unet2D(inp_shape=(None, None, 1)):
    inputs = Input(shape=inp_shape)

    # Three conv pool blocks
    p1, c1 = conv2D_maxpool_block(inputs, 16, 'relu', 'same', False)
    p2, c2 = conv2D_maxpool_block(p1, 32, 'relu', 'same', False)
    p3, c3 = conv2D_maxpool_block(p2, 64, 'relu', 'same', False)
    p4, c4 = conv2D_maxpool_block(p3, 128, 'relu', 'same', False)

    # Fourth conv -- lowest point
    c5 = conv2D_block(p4, 256, 'relu', 'same', False)

    # Three upsampling conv blocks
    cm2 = upsamp_conv2D_block(c5, c4, 128, 'relu', 'same', False)
    cm3 = upsamp_conv2D_block(cm2, c3, 64, 'relu', 'same', False)
    cm4 = upsamp_conv2D_block(cm3, c2, 32, 'relu', 'same', False)
    cm5 = upsamp_conv2D_block(cm4, c1, 16, 'relu', 'same', False)

    # Output
    predictions = Conv2D(1, 1, activation='sigmoid')(cm5)
    model = Model(inputs, predictions)

    return model 


# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[ ]:


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# In[ ]:


# Build model
model2D = build_unet2D(inp_shape=(None, None, 1))

params = {'dim': (1024, 1024),
          'batch_size': 8,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(train_fns[0:8000], annotated, **params)
validation_generator = DataGenerator(train_fns[8000:10712], annotated, **params) 

# Compile model
optimizer = optimizers.Adam(lr = 0.001, epsilon = 0.1)
loss = dice_coef_loss
metrics= [dice_coef]
model2D.compile(optimizer=optimizer, loss=loss, metrics= metrics)

# Fit model
model2D.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=10, verbose=1)


# In[ ]:




