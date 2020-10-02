#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

# libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils

# hyper-parameters
NB_EPOCH = 10
BATCH_SIZE = 128
VERBOSE = 0
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = Adam() # faster
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1) # see ~/.keras/keras.json, this input shape is channels last
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

# data processing
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
NUM_TRAIN, NUM_TEST = train.shape[0], test.shape[0]
y = train.label
train.drop('label', axis = 1, inplace = True)

train = train.values/255
test = test.values/255
train = train.reshape(NUM_TRAIN, IMG_ROWS, IMG_COLS, 1)
test = test.reshape(NUM_TEST, IMG_ROWS, IMG_COLS, 1)
Y = np_utils.to_categorical(y, NB_CLASSES)

# model config
model = Sequential()
model.add(Conv2D(32, kernel_size = 5, padding = 'same', input_shape = INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, kernel_size = 5, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# fit the model
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(train, Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# predict and generate submission
prediction = model.predict_classes(test)
sub = pd.DataFrame({'ImageId': range(1, test.shape[0]+1), 'Label':prediction})
sub.to_csv('submission_cnn_two_hidden_layers.csv', index = False)
