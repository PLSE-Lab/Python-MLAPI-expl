#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[1]:


#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageFile
from IPython.display import display


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[2]:


images = np.load('../input/images.npy')
print(images.shape)
labels = np.load('../input/labels.npy')
print(labels.shape)


# In[ ]:


flipped_images = []
for i, img in enumerate(images):
    im = Image.fromarray(images[i])
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    np.concatenate((images, np.expand_dims(np.array(im),axis=0)), axis=0)


# In[ ]:


img = images
for j in range(0,len(images)//20):
    temp_array = []
    for i in range(0,100):
        im = Image.fromarray(images[j*20+i])
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        temp_array.append(np.array(im))
    img = np.concatenate((img, temp_array), axis=0)
    if(j*100+i == 1999):
        break
        


# In[ ]:


labels = np.concatenate((labels, labels), axis=0)
img.shape


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# In[ ]:


from keras.models import Sequential

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), padding="same",input_shape=(480, 640, 3)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(32, (3, 3), padding="same"),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1),
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error', 'mean_absolute_error','mean_absolute_percentage_error'])

model.summary()


# In[ ]:


model.fit(images, labels, epochs=30, batch_size = 32, validation_split = 0.2, shuffle=True)


# In[ ]:


from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

images = preprocess_input(images)
my_new_model = Sequential()
my_new_model.add(Xception(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.add(Dense(128))
my_new_model.add(Dense(1))

# Say not to train first layer (Xception) model. It is already trained
my_new_model.layers[0].trainable = True

my_new_model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error', 'mean_absolute_error','mean_absolute_percentage_error'])

my_new_model.summary()


# In[ ]:


my_new_model.fit(images, labels, epochs=30, batch_size = 32, validation_split = 0.2, shuffle=True)


# In[3]:


from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras import regularizers

# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False, input_shape = (480,640,3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.4) (x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2) (x)
x = Dense(16, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(1)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers[:35]:
   layer.trainable = False
for layer in model.layers[35:]:
   layer.trainable = True


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error', 'mean_absolute_error','mean_absolute_percentage_error'])

model.summary()


# **Best Result 5.57**
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.4) (x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.2) (x)
# x = Dense(16, activation='relu')(x)
# predictions = Dense(1)(x)
# 
# > trainable layer 40

# In[4]:


from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint

images = preprocess_input(images)
filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(images, labels, epochs=40, batch_size = 32, validation_split = 0.3199, shuffle=True, callbacks=[checkpoint])


# In[5]:


from keras.models import load_model
model = load_model('weights_best.h5')

for layer in model.layers[:40]:
   layer.trainable = False
for layer in model.layers[40:]:
   layer.trainable = True


# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error', 'mean_absolute_error','mean_absolute_percentage_error'])

filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(images, labels, epochs=2, batch_size = 32, validation_split = 0.3199, shuffle=True, callbacks=[checkpoint])


# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# 
# SVG(model_to_dot(model).create(prog='dot', format='svg'))

# In[ ]:


model.layers[132]


# In[ ]:




