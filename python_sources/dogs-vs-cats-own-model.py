#!/usr/bin/env python
# coding: utf-8

# #Image Generator

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(len(os.listdir("../input/train")))

# Any results you write to the current directory are saved as output.


# In[2]:


os.mkdir("modifiedtrain")
os.mkdir("modifiedtrain/cat")
os.mkdir("modifiedtrain/dog")


# In[3]:


os.listdir("modifiedtrain")


# In[ ]:


from shutil import copyfile
for file in os.listdir("../input/train"):
    name=file.split('.')[0]
    filename="../input/train/"+file
    if name=='cat':
        copyfile(filename,"modifiedtrain/cat/"+file)
    elif name=='dog':
        copyfile(filename,"modifiedtrain/dog/"+file)
    
    


# In[ ]:


os.listdir("modifiedtrain/dog/")


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
from PIL import Image
image = Image.open('modifiedtrain/dog/dog.411.jpg')
plt.imshow(image)
plt.show()


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen=ImageDataGenerator(rescale=1./255)


# In[ ]:


train_generator=train_datagen.flow_from_directory("modifiedtrain/",batch_size=20,target_size=(150,150),
                                                  class_mode='binary')


# In[ ]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])


# 

# In[ ]:


history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=15)


# In[ ]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[ ]:


train_generator=train_datagen.flow_from_directory("modifiedtrain/",batch_size=20,target_size=(150,150),
                                                  class_mode='binary')


# In[ ]:


history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=5)


# In[ ]:


import os

from tensorflow.keras import layers
from tensorflow.keras import Model
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
  
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (256, 256, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[ ]:


train_generator=train_datagen.flow_from_directory("modifiedtrain/",batch_size=20,target_size=(256,256),
                                                  class_mode='binary')


# In[ ]:


history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=5)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.legend(loc=0)
plt.figure()


# In[ ]:


plt.plot(epochs, loss, 'b', label='Training loss')
plt.legend(loc=0)
plt.figure()

