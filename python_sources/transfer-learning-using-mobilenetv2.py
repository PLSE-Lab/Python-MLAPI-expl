#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Importing libaries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,models,applications,preprocessing,optimizers
import cv2


# Loading training data  and test data into lists

# In[ ]:


train_data = []
train_labels = []
test_data = []
test_labels = []
IMG_SIZE = 160
def get_images(path):
    cfiles = os.listdir(os.path.join(path,'cats'))
    dfiles = os.listdir(os.path.join(path,'dogs'))
    data = []
    labels = []
    for i in dfiles:
        try:
            imgpath = os.path.join(path,'dogs') 
            img = cv2.imread(os.path.join(imgpath,i))
            img = tf.image.resize(img,(IMG_SIZE,IMG_SIZE))
            data.append(img)
            labels.append(1)
        except:
            pass
    for i in cfiles:
        try:
            imgpath = os.path.join(path,'cats') 
            img = cv2.imread(os.path.join(imgpath,i))
            img = tf.image.resize(img,(IMG_SIZE,IMG_SIZE))
            data.append(img)
            labels.append(0)
        except:
            pass
    return data, labels

train_data, train_labels = get_images('/kaggle/input/cat-and-dog/training_set/training_set')
test_data, test_labels = get_images('/kaggle/input/cat-and-dog/test_set/test_set')


# Checking their length

# In[ ]:


len(train_data),len(train_labels),len(test_data),len(test_labels)


# Converting training data and test data into tensorflow dataset objects

# In[ ]:


train_data = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_data,test_labels))


# Formatting the data

# In[ ]:


def format_example(image,label):
    image = tf.cast(image,dtype = tf.float32)
    image = (image / 255) - 1
    image = tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    return image, label
train_data = train_data.map(format_example)
test_data = test_data.map(format_example)


# Making the data into batches and shuffling them

# In[ ]:


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 2000
train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = test_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


# Intializing the mobilenet v2 with pretrained weights

# In[ ]:


IMG_SHAPE = (IMG_SIZE,IMG_SIZE,3)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
        include_top = False,weights = '../input/mobilenet-v2-keras-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')

base_model.trainable = False
base_model.summary()


# Checking training batches shape

# In[ ]:


for image_batch, label_batch in train_data.take(1):
    pass
print(image_batch.shape)


# Checking the output shape from the basemodel(mobilenet v2)

# In[ ]:


print(base_model(image_batch).shape)


# adding global average layer to convert output from the base model feedable to hidden layer

# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
print(global_average_layer(base_model(image_batch)).shape)


# Hidden layer to train

# In[ ]:


hidden_layer = tf.keras.layers.Dense(32,activation = 'relu')
print(hidden_layer(global_average_layer(base_model(image_batch))).shape)


# prediction layer to predict

# In[ ]:


prediction_layer = tf.keras.layers.Dense(1)
print(prediction_layer(hidden_layer(global_average_layer(base_model(image_batch)))).shape)


# Initializing the  model with the above layers

# In[ ]:


model = tf.keras.Sequential([
     base_model,
     global_average_layer,
     hidden_layer,
     prediction_layer
])


# fixing the learning rate and compiling the model

# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate),
              loss = 'binary_crossentropy', metrics = ['accuracy'])


# summary

# In[ ]:


model.summary()


# adding model checkpoint callback

# In[ ]:


callbacks = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# calculating the steps per epochs

# In[ ]:


num_train = 2000
num_test = 1000
initial_epochs = 20
steps_per_epochs = round(num_train) //  BATCH_SIZE
validation_steps = 4

loss0, accuracy0 = model.evaluate(test_data, steps = validation_steps)


# Training the model

# In[ ]:


history = model.fit(train_data,epochs = 10,callbacks = [callbacks],validation_data = test_data)


# loading weights from the saved best model

# In[ ]:


model.load_weights('best_model.h5')


# predicting from the best model

# In[ ]:


model.evaluate(test_data)


# we got 76% accuracy.
