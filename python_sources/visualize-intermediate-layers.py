#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import keras
from PIL import Image
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image  import ImageDataGenerator
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras import models
from keras.applications.vgg16 import decode_predictions
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D


# In[3]:


pil_im=Image.open("../input/1a146bf4-5696-4e19-842e-9e0e1ac7ac6d___RS_LB 5023.JPG")
imshow(np.asarray(pil_im))


# In[4]:


from keras.models import load_model
classifier=load_model("../input/alexnet_20.h5")


# In[5]:


img=np.array(pil_im)
image = np.expand_dims(img, axis=0)


# In[6]:


classifier.summary()


# In[7]:


layer_outputs = [layer.output for layer in classifier.layers[:18]] 
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)


# In[8]:


activations = activation_model.predict(image)


# In[9]:


len(activations)


# In[10]:


first_layer_activation = activations[2]
print(first_layer_activation.shape)


# In[11]:


pil_im=Image.open("../input/1a146bf4-5696-4e19-842e-9e0e1ac7ac6d___RS_LB 5023.JPG")
imshow(np.asarray(pil_im))


# In[12]:


for i in range(18):
    plt.matshow(activations[i][0, :, :, 4], cmap='viridis')


# In[ ]:


layer_names = []
for layer in classifier.layers[:17]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 10

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[ ]:




