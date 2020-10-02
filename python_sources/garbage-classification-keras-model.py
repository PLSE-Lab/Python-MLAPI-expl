#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adadelta
from sklearn.datasets import load_files
from skmultilearn.model_selection import iterative_train_test_split


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 strides=2,
                 input_shape=(64,64, 3),
                 activation ='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 6, activation = 'softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['sparse_categorical_accuracy'])


# In[ ]:


train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


test_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.1)


# In[ ]:


X_train = train_datagen.flow_from_directory('/kaggle/input/garbage-classification/garbage classification/Garbage classification/',
                                    target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')

X_test = test_datagen.flow_from_directory('/kaggle/input/garbage-classification/garbage classification/Garbage classification/',
                          target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'sparse')


# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)


# In[ ]:


cd = [es, mc]


# In[ ]:



model.fit_generator(
    X_train,
    steps_per_epoch=X_train.samples//X_train.batch_size,
    epochs=1000,
    callbacks=cd,
    validation_data=X_test,
    validation_steps=X_test.samples//X_test.batch_size)


# In[ ]:


model.save("new_model.h5")


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (64, 64, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


# In[ ]:


Image.open('/kaggle/input/more-test-images/IMG-7974.JPG')


# In[ ]:


Image.open('/kaggle/input/more-test-images/IMG-7976.JPG')


# In[ ]:


image1 = load('/kaggle/input/more-test-images/IMG-7974.JPG')
image2 = load('/kaggle/input/more-test-images/IMG-7976.JPG')
image3 = load('/kaggle/input/more-test-images/images.jpeg')
image4 = load('/kaggle/input/more-test-images/Screen-Shot-2019-05-22-at-10.27.26-AM (1).jpg')


# In[ ]:


prediction1 = model.predict_classes(image1)
print(prediction1)


# In[ ]:


prediction2 = model.predict_classes(image2)
print(prediction2)


# In[ ]:


prediction3 = model.predict_classes(image3)
print(prediction3)


# In[ ]:


prediction4 = model.predict_classes(image4)
print(prediction4)


# In[ ]:


X_train.class_indices


# In[ ]:


def prediction(result):
    if result == 0:
        prediction = "metal"
    elif result == 1:
        prediction = "paper"
    elif result == 2:
        prediction = "plastic"
    elif result == 3:
        prediction = "glass"
    elif result == 4:
        prediction = "trash"
    elif result == 5:
        prediction = "cardboard"
    else:
        prediction = "error"
    return prediction


# In[ ]:


prediction(prediction1)


# In[ ]:


prediction(prediction2)


# In[ ]:


prediction(prediction3)


# In[ ]:


prediction(prediction4)


# In[ ]:


def total(filename):
    image = load(filename)
    result = model.predict_classes(image)
    return prediction(result)


# In[ ]:


total('/kaggle/input/more-test-images/IMG-7974.JPG')


# In[ ]:


total('/kaggle/input/more-test-images/IMG-7976.JPG')


# In[ ]:


total('/kaggle/input/more-test-images/images.jpeg')


# In[ ]:


total('/kaggle/input/more-test-images/Screen-Shot-2019-05-22-at-10.27.26-AM (1).jpg')

