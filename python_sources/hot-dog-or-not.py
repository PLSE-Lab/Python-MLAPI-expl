#!/usr/bin/env python
# coding: utf-8

# # Hot Dog or Not?
#    The Hot dog or not is a binary classification problem where we need predict whether the image contains hot dog or not.  In this notebook, **TensorFlow** is used to create the Neural Network.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# the **generator()** function creates an ImageDataGenerator object and return it to the calling function.  To increase the accuracy of the model, data augmentation is used.

# In[ ]:


def generator():
    data_generator=tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    return data_generator


# Here, training and testing ImageDataGenerator were created by calling the generator function.

# In[ ]:


train_generator=generator()
test_generator=generator()


# In[ ]:


def data_flow(path,generator):
    data_flow=train_generator.flow_from_directory(path,batch_size=100,class_mode='binary')
    return data_flow;


# In[ ]:


train_dir='../input/hot-dog/hotdog/train'
train_data_generator=data_flow(train_dir,train_generator)


# In[ ]:


test_dir='../input/hot-dog/hotdog/test'
test_data_generator=data_flow(test_dir,train_generator)


# # Model Creation
#    The Convolutional Neural network is created below.  This neural network consists of 3 convolution layer and 3 neural network layers.  Dropout is being implemented in the network to eliminate some neurons.  Here, the model eliminates 40% of the neurons with Dropout.
#    The output layer has a single neuron which tells whether the image contains hot dog or not.
# ![](https://cdn-images-1.medium.com/max/1000/1*uxpH46OpTIj63j1MKQ-T2Q.png)  

# In[ ]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])


# # MODEL COMPILATION
#    The model is compiled by using the loss function as **binary cross entropy** and the optimizer used here is **RMSprop**.

# In[ ]:


model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=0.001),metrics=['acc'])


# # TRAINING
#   The network is trained with the training data for upto 15 epochs.

# In[ ]:


model_training=model.fit_generator(train_data_generator,
                                   epochs=15,
                                   validation_data=test_data_generator,
                                   validation_steps=8)


# The Network gave an accuracy of 82% with training data and 80% with testing data.

# In[ ]:


accuracy=model_training.history['acc']
validation_acc=model_training.history['val_acc']
loss=model_training.history['loss']
validation_loss=model_training.history['val_loss']


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(list(range(15)),accuracy,label="Accuracy")
plt.plot(list(range(15)),validation_acc,label="Validation Accracy")
plt.xlabel("EPOCHS")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()


# In[ ]:


plt.plot(list(range(15)),loss,label="Loss")
plt.plot(list(range(15)),validation_loss,label="Validation Loss")
plt.xlabel("EPOCHS")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()


# In[ ]:


plt.bar(list(range(15)),accuracy)
plt.xlabel("EPOCHS")
plt.ylabel("Accuracy")
plt.show()


# # SAVE MODEL

# In[ ]:


model.save('model')


# In[ ]:


import shutil

zip_name = 'trained_model'
directory_name = 'model'

shutil.make_archive(zip_name, 'zip', directory_name)


# In[ ]:




