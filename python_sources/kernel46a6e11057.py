#building the convolutional neural network

import numpy as np
import pandas as pd

import os
print(os.listdir("../input/convolutional-neural-networks/Convolutional_Neural_Networks/dataset"))#importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

#initialising the CNN
classifier = Sequential()

#step1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#adding additional convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPool2D(pool_size=(2, 2)))
#step2 pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

#step 3 flattening
classifier.add(Flatten())

#step 4 full connection
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#compiling the cnn
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the cnn to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('../input/convolutional-neural-networks/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('../input/convolutional-neural-networks/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)