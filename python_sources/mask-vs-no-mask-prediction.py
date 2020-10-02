#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import cv2


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from keras.applications.vgg16 import VGG16


# In[ ]:


import keras
from keras.models import Input, Model, Sequential
from keras.layers import Dense, Flatten, Conv1D


# In[ ]:


from keras import backend as K


# # read the data

# In[ ]:


train_csv = pd.read_csv('../input/train.csv')


# In[ ]:


train_csv.head()


# In[ ]:


ids = train_csv.id.tolist()


# In[ ]:


images = np.array(list(map(lambda x: cv2.imread('../input/train/images/'+ x + '.png'), ids)))


# In[ ]:


rle_mask = train_csv.rle_mask.isna()


# # Build binary classifier (Nan or not)

# # Split the data

# In[ ]:


images.shape


# In[ ]:


images = np.array(list(map(lambda x: cv2.resize(x, (299, 299)), images)))


# In[ ]:


images.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(images, rle_mask, test_size=0.2, random_state=42)


# # Build the model

# **I will use pretrained VGG, but will skip the fully connected layers and last pooling and  last 2 convolution layers as the data contains basic shapes (lines, circules, ..) not complecated shapes like cars and faces.**

# # Part of VGG

# In[ ]:


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3))


# In[ ]:


vgg.summary()


# In[ ]:


# input = Input(shape=(299, 299, 3))


layers = dict([(layer.name, layer) for layer in vgg.layers])

vgg_top = layers['block5_conv2'].output

x = Flatten(name='flatten')(vgg_top)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(512, activation='relu', name='fc3')(x)
x = Dense(1, activation='sigmoid', name='predictions')(x)

my_model = Model(input=vgg.input, output=x)


# In[ ]:


my_model.summary()


# In[ ]:


for layer in vgg.layers:
    layer.trainable = False


# In[ ]:


sgd = keras.optimizers.SGD(lr=0.00001, momentum=0.9, nesterov=True)
my_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=45,
    horizontal_flip=True,
    vertical_flip=True)


datagen.fit(X_train)


# In[ ]:


# fits the model on batches with real-time data augmentation:
my_model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=50)


# In[ ]:


my_model.save_weights('model_50.h5')


# In[ ]:


my_model.evaluate(X_test, y_test)


# the learning rate seems so slow

# In[ ]:


K.eval(my_model.optimizer.lr.assign(0.0001))


# In[ ]:


# fits the model on batches with real-time data augmentation:
my_model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=20)


# In[ ]:


my_model.evaluate(X_test, y_test)


# In[ ]:




