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

# example of using the vgg16 model as a feature extraction model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential


# load model without classifier layers
model = VGG16(include_top=False, input_shape=(64, 64, 3))
model.summary()
model2 = model
'''
for i in range(0,12):
#    print(model.get_layer(index = i))
    model2.layers.pop()
'''
#model2.summary()
#model2 = model.layers.pop(2)

classifier = Sequential()

classifier.add(model)

classifier.add(Flatten())

#Step 4 = Full Connection
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set  = train_datagen.flow_from_directory('../input/training_set/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')


test_set = test_datagen.flow_from_directory(
        '../input/test_set/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')



classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/8,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 2000/2)

