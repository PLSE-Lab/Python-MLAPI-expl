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
print(os.listdir("../input/kerneleb49bdddc6/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from keras.applications.densenet import DenseNet201
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


# In[ ]:


trgen = image.ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    horizontal_flip=True)
valgen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
    )


# In[ ]:


base_model = DenseNet201(weights=None, include_top=False)


# In[ ]:


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)


# In[ ]:


model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
    #layer.trainable = False
import keras
op = keras.optimizers.SGD(lr=0.1, momentum=0.9)
import tensorflow as tf
def logloss(y, y_):
    return tf.losses.log_loss(y,y_)
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=[logloss])


# In[ ]:


X_train = -1+np.load('../input/kerneleb49bdddc6/X_train.npy')
X_val = -1+np.load('../input/kerneleb49bdddc6/X_val.npy')
y_train = np.load('../input/kerneleb49bdddc6/y_train.npy')
y_val = np.load('../input/kerneleb49bdddc6/y_val.npy')


# In[ ]:


trgen.fit(X_val)
#valgen.fit(X_val)


# In[ ]:


#X_val -= valgen.mean
# Apply featurewise_std_normalization to test-data with statistics from train data
#X_val /= (valgen.std + K.epsilon())


# In[ ]:


from keras.callbacks import ModelCheckpoint
import keras
get_ipython().system('mkdir Graph')
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
filepath="model.hdf5"
ch = ModelCheckpoint(filepath, monitor='val_logloss',verbose=1, save_best_only=True, mode='min')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1, cooldown=1)


# In[ ]:


#model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)
#model.fit_generator(trgen.flow(X_train, y_train, batch_size=128), steps_per_epoch=len(X_train) / 128, epochs=2, validation_data=(X_val, y_val), callbacks=[ch, tbCallBack])


# for layer in model.layers[:0]:
#     layer.trainable = False
# for layer in model.layers[0:]:
#     layer.trainable = True
#     
# model.compile(optimizer='adam', loss=logloss)

# In[ ]:


#trgen.fit(X_train, y_train)
model.fit_generator(trgen.flow(X_train, y_train, batch_size=16), steps_per_epoch=len(X_train) / 16, epochs=50, validation_data=(X_val, y_val), 
                    callbacks=[ch, tbCallBack], class_weight = 'auto')


# In[ ]:


model.save('mf.h5')
from keras.models import load_model
model = load_model('model.hdf5', 
                  custom_objects={'logloss':logloss})


# In[ ]:


model.evaluate(X_val,y_val)


# In[ ]:


model.predict(X_val).sum(axis=0)


# In[ ]:


#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc', logloss])
#model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

