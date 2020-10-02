# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../"))

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau


input_shape=(64,64,3)
num_classes=12

classifier = Sequential()

#1.Convolution
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

classifier.add(BatchNormalization())

#2.Pooling -for reducing size of feature map
#classifier.add(MaxPool2D(pool_size=(2,2) ))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPool2D(2,2) )

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2) ))


#3.Flattening - pool to vector(input for ANN)
classifier.add(Dropout(0.5))
classifier.add(Flatten())

#4.ANN creation

#classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(32,activation='relu'))
classifier.add(Dense(num_classes, activation='softmax'))

#compiling CNN
classifier.compile(optimizer='adam' ,loss = 'categorical_crossentropy',metrics=['accuracy'])

# We'll stop training if no improvement after some epochs
earlystopper1 = EarlyStopping(monitor='loss', patience=10, verbose=1)

# Save the best model during the traning
checkpointer1 = ModelCheckpoint('best_model1.h1'
                                ,monitor='val_acc'
                                ,verbose=1
                                ,save_best_only=True
                                ,save_weights_only=False)

#fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        '../input/nonsegmentedv2/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_steps = 100,
        epochs=20,
        validation_data = validation_generator, 
        callbacks=[earlystopper1, checkpointer1])

# Any results you write to the current directory are saved as output.