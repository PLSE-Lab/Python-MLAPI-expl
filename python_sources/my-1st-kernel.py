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


pwd


# In[ ]:


cd /kaggle/input


# In[ ]:


cd cats-dataset/


# In[ ]:


ls


# In[ ]:


cd Cats/


# In[ ]:


ls


# In[ ]:


import pandas as pd


# In[ ]:


import cv2


# In[ ]:


pwd


# In[ ]:


import numpy as np


# In[ ]:


import os


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


a=plt.imread("/kaggle/input/cats-dataset/Cats/images (1).jpg")


# In[ ]:


plt.imshow(a)


# In[ ]:


img = cv2.imread("/kaggle/input/cats-dataset/Cats/images (1).jpg", cv2.IMREAD_UNCHANGED)

#get dimensions of image
dimensions=img.shape

#height, width, number of channels in image
height = img.shape[0]
width  = img.shape[1]
channels = img.shape[2]

print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image width        : ',width)
print('Number of Channels : ',channels)


# In[ ]:


plt.imshow(img)


# In[ ]:


import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    
    shear_range = 0.02,
    fill_mode = 'reflect',
    zoom_range = [0.1,0.4],
    brightness_range = [0.15,0.30],
    cval = 1,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    validation_split = 0.2,
)

valid_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = '/kaggle/input/',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training')
validation_generator = train_datagen.flow_from_directory(
    directory = '/kaggle/input/',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation')


# In[ ]:


from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, LeakyReLU
from keras.optimizers import Adam, SGD, RMSprop
from keras import applications, metrics
from keras import backend as K


# In[ ]:


model = applications.VGG16(weights = "imagenet", include_top = False, input_shape = (224,224,3))


# In[ ]:


model.summary()


# In[ ]:


for layer in model.layers[:]:
    layer.trainable = False
    
    #Adding Custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(300, activation = "relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(300, activation = "relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation = "sigmoid")(x)
    
    #Creating the final model
    model_final = Model(input = model.input, output = predictions)
    model_final.compile(loss = "binary_crossentropy",optimizer = SGD(lr = 1e-3),metrics = ['accuracy'])
    


# In[ ]:


model_final.summary()


# In[ ]:


model_history = model_final.fit_generator(
train_generator,
steps_per_epoch = 100/1000,
validation_data = validation_generator,
validation_steps = 100/1000,
epochs = 50,
verbose = 1)


# In[ ]:


fig = plt.figure(figsize = (12,8))
plt.plot(model_history.history['loss'],'blue')
plt.plot(model_history.history['val_loss'],'orange')
plt.xticks(np.arange(0,50.5))
plt.yticks(np.arange(0,1,.1))
plt.rcParams['figure.figsize'] = (10,10)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs validation Accuracy for Pets_Classification")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()

plt.figure(1)
plt.plot(model_history.history['loss'],'blue')
plt.plot(model_history.history['val_loss'],'orange')
plt.xticks(np.arange(0,50.5))
plt.yticks(np.arange(0,1,.1))
plt.rcParams['figure.figsize'] = (10,10)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs validation Accuracy for Pets_Classification")
plt.grid(True)
plt.gray()
plt.legend(['train','validation'])
plt.show()


# In[ ]:




