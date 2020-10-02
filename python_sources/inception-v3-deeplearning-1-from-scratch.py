# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import models, optimizers

import os

print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.
train_directory ='../input/train/train'
test_directory ='../input/test/test'

train_datagen = ImageDataGenerator(
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')
test_datagen   = ImageDataGenerator(
        rescale=1./255)
          
train_generator = test_datagen.flow_from_directory(
                  train_directory,
                  target_size=(299,299),
                  shuffle=False,
                  batch_size=16)
train_classes = train_generator.classes
print(train_classes)
test_generator = train_datagen.flow_from_directory(
                  test_directory,
                  target_size=(299,299),
                  shuffle=False,
                  batch_size=16)

pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
#pretrained_model.summary()

#img_width, img_height = 299, 299
model = Sequential()
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dense(67, activation= 'softmax'))

model.summary()

print ('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
pretrained_model.trainable=False
print ('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))


# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', # or categorical_crossentropy
              optimizer=optimizers.RMSprop(lr=0.0001),# or adagrad
              metrics=['accuracy'])


    
batch_size=32              
history = model.fit_generator(
          train_generator,
          steps_per_epoch=4000// batch_size,#2000
          epochs=30, #50
          validation_data=test_generator,
          validation_steps=1000// batch_size)
          
          

          
          
          
          

