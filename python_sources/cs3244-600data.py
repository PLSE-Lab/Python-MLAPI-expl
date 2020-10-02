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


import numpy as np 
import pandas as pd 
import os

import tensorflow.keras as tk
from keras import optimizers
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam
from keras.preprocessing.image import load_img, save_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Activation,     BatchNormalization, GlobalAveragePooling2D
from keras_applications.resnet50 import preprocess_input


# In[ ]:


train_path='/kaggle/input/cs3244-xiuping-600-oversampled/Artists600duplicates/train'


# In[ ]:


def select_image(image_path, artists_top):
    selected_paints = []
    for name in artists_top['name'].values:
        for paints in os.listdir(image_path):
            if name in paints:
                selected_paints.append(preprocess_image(image_path+paints))
    print("lengths is ", len(selected_paints))
    return selected_paints
def select_name(artists_top):
    selected_names = artists_top['name'].str.replace(' ', '_').values.tolist()
    return selected_names

def read_data (csv_path):
    artists = pd.read_csv(csv_path)
    return artists

def select_artists(artists):
    artists = artists.sort_values(by=['paintings'], ascending=False)
    artists_top = artists[artists['paintings'] >= 0].reset_index()
    artists_top = artists_top[['name', 'paintings']]
  #  print(artists_top)
    return artists_top


# In[ ]:


artists = read_data('/kaggle/input/cs3244-artists/artists.csv')

artists_top = select_artists(artists)
selected_names = os.listdir(train_path)

batch_size = 16
RESNET50_POOLING_AVERAGE = 'avg'
train_input_shape = (224, 224, 3)
NUM_EPOCH = 30
NUM_CLASSES = artists_top.shape[0]


# In[ ]:





# In[ ]:


data_generator = ImageDataGenerator(validation_split=0.2)

path = '/kaggle/input/data/'

train_generator = data_generator.flow_from_directory(train_path, target_size=train_input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset= 'training',
        shuffle=True,
        classes = selected_names)

validation_generator = data_generator.flow_from_directory(train_path, target_size=train_input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'validation',
        shuffle=True,
        classes= selected_names)


# In[ ]:


print(len(set(train_generator.classes.tolist())))
print(len(set(validation_generator.classes.tolist())))


# In[ ]:


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model = Sequential()
model.add(ResNet50(weights = 'imagenet', include_top = False, input_shape = train_input_shape))

for layer in model.layers:
    layer.trainable = True

X = model.output
X = Flatten()(X)


X = Dense(512, kernel_initializer='he_uniform')(X)
X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output = Dense(NUM_CLASSES, activation='softmax')(X)
model = Model(inputs=model.input, outputs=output)

for layer in model.layers[50:]:
    layer.trainable = True


print(model.summary())

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


EARLY_STOP_PATIENCE = 3

file_path = '/kaggle/input/cs3244-noisy-trained-weight/Xiuping600model.hd5'

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

fit_history = model.fit_generator(
        generator= train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs = NUM_EPOCH,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        shuffle=True,
        callbacks=[check_point, early_stop, reduce_lr]
)


# In[ ]:



plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.savefig('performance.png')
plt.show()

