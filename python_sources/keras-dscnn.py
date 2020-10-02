#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com

# libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


# hyper-parameters
BATCH_SIZE = 128
NB_CLASSES = 10 # number of outputs = number of digits
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1) # see ~/.keras/keras.json, this input shape is channels last
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
CALLBACKS = [
    ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 10, min_lr = 0.00002, verbose = 1),
    EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
    ]

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

# train validation split
X_train, X_valid, y_train, y_valid = train_test_split(train, Y, test_size = 0.3, random_state = 1106)


def build_model():
    # model config
    inputs = Input(shape=INPUT_SHAPE)
    
    img_model = Conv2D(filters = 32, kernel_size = 3, strides =  1, padding = 'same')(inputs)
    img_model = BatchNormalization()(img_model)
    img_model = Activation('relu')(img_model)
    img_model = Dropout(0.2)(img_model)

    for i in range(3):
        img_model = SeparableConv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(img_model)
        img_model = BatchNormalization()(img_model)
        img_model = Activation('relu')(img_model)
        img_model = Dropout(0.2)(img_model)    

    global_avg_pool = GlobalAveragePooling2D()(img_model)
    outputs = Dense(NB_CLASSES, activation = "softmax")(global_avg_pool)
    outputs = Dropout(0.1)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
    
# augumented the input data
datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip = False  # randomly flip images
    )

datagen.fit(train)

model = build_model()
# fit the model with augumented data
history = model.fit_generator(
        generator = datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
        steps_per_epoch = len(X_train) / BATCH_SIZE,
        epochs = 10,
        verbose = 2,
        validation_data = datagen.flow(X_valid, y_valid, batch_size = BATCH_SIZE),
        validation_steps = len(X_valid) / BATCH_SIZE,
        callbacks = CALLBACKS
    )

# predict and generate submission
prediction = model.predict(test).argmax(axis = 1)
sub = pd.DataFrame({'ImageId': range(1, test.shape[0]+1), 'Label':prediction})
sub.to_csv('submission_cnn_with_dropout_early_stopping_and_augmented_data.csv', index = False) # 0.99329