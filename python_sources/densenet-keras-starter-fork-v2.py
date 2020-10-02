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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Resources:
# 
# https://www.kaggle.com/xhlulu/aptos-2019-densenet-keras-starter

# # Imports

# In[ ]:


import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
import tensorflow as tf
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed for reproducibility.
np.random.seed(2019)
tf.set_random_seed(2019)


# # Target & ID Loading

# In[ ]:


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# # Image Loading & Pre-processing

# In[ ]:


def preprocess_image(image_path, desired_size=224):
    im = Image.open(image_path)
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
    
    return im


# In[ ]:


# get the number of training images from the target\id dataset
N = train_df.shape[0]
# create an empty matrix for storing the images
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

# loop through the images from the images ids from the target\id dataset
# then grab the cooresponding image from disk, pre-process, and store in matrix in memory
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


# In[ ]:


# do the same thing as the last cell but on the test\holdout set

N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(
        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'
    )


# In[ ]:


# pre-processing the target (i.e. one-hot encoding the target)
y_train = pd.get_dummies(train_df['diagnosis']).values

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[ ]:


# Further target pre-processing

# Instead of predicting a single label, we will change our target to be a multilabel problem; 
# i.e., if the target is a certain class, then it encompasses all the classes before it. 
# E.g. encoding a class 4 retinopathy would usually be [0, 0, 0, 1], 
# but in our case we will predict [1, 1, 1, 1]. For more details, 
# please check out Lex's kernel.

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# # Train & Validation Split

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.50, 
    random_state=2019
)


# # Create Image Augmentation Generator

# In[ ]:


BATCH_SIZE = 13

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# Using original generator
data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)


# # Create Model

# In[ ]:


densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)


# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.80))
    model.add(layers.Dense(5, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00010509613402110064),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# # Train Model

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# In[ ]:


kappa_metrics = Metrics()

#history = model.fit_generator(
#    data_generator,
#    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
#    epochs=17,
#    validation_data=(x_val, y_val),
#    callbacks=[kappa_metrics]
#)

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val)
)


# # Training Plots

# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


#plt.plot(kappa_metrics.val_kappas)


# # Submission

# In[ ]:


y_test = model.predict(x_test)
y_test


# In[ ]:


y_test = y_test > 0.37757874193797547
y_test


# In[ ]:


y_test.astype(int).sum(axis=1)


# In[ ]:


y_test.astype(int).sum(axis=1) - 1


# In[ ]:


y_test = y_test.astype(int).sum(axis=1) - 1
y_test


# In[ ]:


test_df['diagnosis'] = y_test
test_df.to_csv('submission.csv',index=False)

