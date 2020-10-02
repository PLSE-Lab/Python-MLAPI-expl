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


import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


train_path = "/kaggle/input/cat-and-dog/training_set/training_set"
test_path = "/kaggle/input/cat-and-dog/test_set/test_set"


# In[ ]:


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


# In[ ]:


train_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
                train_path ,target_size=(IMG_SIZE,IMG_SIZE),batch_size=24,class_mode='categorical')

test_batches = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
                test_path ,target_size=(IMG_SIZE,IMG_SIZE),batch_size=24,class_mode='categorical')


# In[ ]:


base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
base_model.trainable = True


# In[ ]:


feature_batch = base_model.output

global_average_layer = keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = keras.layers.Dense(2)
prediction_batch = prediction_layer(feature_batch_average)


# In[ ]:


fine_tune_at = 100


# In[ ]:


for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[ ]:


model = keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


len(model.trainable_variables)


# In[ ]:


history = model.fit_generator(train_batches,
                         steps_per_epoch=24,
                         epochs=10,
                         validation_data=test_batches,
                         validation_steps=200)


# In[ ]:


loss0,accuracy0 = model.evaluate(test_batches, steps = 20)


# In[ ]:


accuracy0


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.show()


# In[ ]:


plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

