#!/usr/bin/env python
# coding: utf-8

# The goal is to train a model which can determine sign language using image augumentation, callbacks in keras

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


import tensorflow as tf
tf.__version__


# Let's try to see if we can read the data from  csv file and get the image pixel and label values

# In[ ]:


df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.iloc[:,1:].shape


# In[ ]:


labels = np.asarray(df.iloc[:,:1]).astype(np.float32)


# In[ ]:


labels.shape


# In[ ]:


images = np.asarray(df.iloc[:,1:]).astype(np.float32)
images.shape


# In[ ]:


images = images.reshape((-1,28,28))
images.shape


# In[ ]:


images = np.expand_dims(images, axis=-1)
images.shape


# plot an image to verify

# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(np.squeeze(images[1], axis=-1), cmap='binary')
print(chr(labels[1] + 65))
plt.show()


# Defining a function for easy code reuse

# In[ ]:


def read_data(filename):
    df = pd.read_csv(filename)
    labels = np.asarray(df.iloc[:,:1]).astype(np.float32)
    images = np.asarray(df.iloc[:,1:]).astype(np.float32)
    images = images.reshape((-1,28,28))
    images = np.expand_dims(images, axis=-1)
    
    return images, labels


# In[ ]:


train_filename = '../input/sign-language-mnist/sign_mnist_train.csv'
test_filename = '../input/sign-language-mnist/sign_mnist_test.csv'


# In[ ]:


train_images, train_labels = read_data(train_filename)
test_images, test_labels = read_data(test_filename)


# In[ ]:


train_images.shape, train_labels.shape


# In[ ]:


test_images.shape, test_labels.shape


# Create an ImageDataGenerator for data augumentation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255.,
                                  rotation_range=40,
                                  width_shift_range=0.25,
                                  height_shift_range=0.25,
                                  shear_range=0.2,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255.)


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(26, activation='softmax')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Define callbacks function

# In[ ]:


class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.8:
            print('Model crossed threshold accuracy, hence stopped training')
            self.mode.stop_training = True
            
callbacks = custom_callback()


# In[ ]:


history = model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=32),
                             steps_per_epoch=len(train_images)/32,
                             epochs=20,
                             validation_data=test_datagen.flow(test_images, test_labels, batch_size=32),
                             validation_steps=len(test_images)/32,
                             callbacks=[callbacks])


# It is clearly underfitting. Plotting the metrics for easy visualiztion.

# In[ ]:


history.params


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:




