#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# chest-xray-pneumonia
get_ipython().run_line_magic('reset', '-f')
import numpy as np  
import pandas as pd  
import os


# In[ ]:


# libraries for image processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Libraries for building sequential CNN model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

# Save CNN model configuration
from tensorflow.keras.models import model_from_json

# For ROC plotting
import matplotlib.pyplot as plt


# In[ ]:


# Adjusting the Dimensions of images
img_width, img_height = 150, 150


# In[ ]:


#Data folder containing all training images
train_data_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train"


# In[ ]:


# Total number of training images
nb_train_samples = 5216 


# In[ ]:


#Data folder containing all validation images
validation_data_dir = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"


# In[ ]:


# Total no of validation samples that should be generated
nb_validation_samples = 16


# In[ ]:


#Batch size to train at one go:
batch_size = 25


# In[ ]:


#Epochs of training:
epochs = 10


# In[ ]:


# No of test samples
test_generator_samples = 300


# In[ ]:


#For test data, batch size should be
test_batch_size = 8 


# In[ ]:


# keras backend configuration values, as:
K.image_data_format()
K.backend()


# In[ ]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:                                           
    input_shape = (img_width, img_height, 3)


# In[ ]:


#Create convnet model
model = Sequential()


# In[ ]:


# 2D Convolution layers:

model.add(Conv2D(
                filters=32,                       # For every filter there is set of weights                                              # For each filter, one bias. So total bias = 32
                kernel_size=(3, 3),               # For each filter there are 3*3=9 kernel_weights, it is always square means (x, x)
                strides = (1,1),                  # So output shape will be 148 X 148 (W-F+1).
                                                  # Default strides is 1 only
                input_shape=input_shape,          # (150,150,3)
                use_bias=True,                     # Default value is True
                padding='same',                   # 'valid' => No padding. This is default.
                name="Ist_conv_layer"
                )
                 )


# In[ ]:


model.add(Activation('relu'))


# In[ ]:


model.add(Conv2D(
                filters=16,                       # For every filter there is set of weights
                                                    # For each filter, one bias. So total bias = 32
                kernel_size=(3, 3),               # For each filter there are 3*3=9 kernel_weights, it is always square means (x, x)
                strides = (1,1),                  # So output shape will be 148 X 148 (W-F+1).
                                                    # Default strides is 1 only
                use_bias=True,                     # Default value is True
                padding='same',                   # 'valid' => No padding. This is default.
                name="IInd_conv_layer"
                )
                 )


# In[ ]:


model.add(Activation('relu'))


# In[ ]:


model.summary()


# In[ ]:


# Pool_size:
model.add(MaxPool2D(pool_size=(2, 2)))


# In[ ]:


# Flattens the input.

model.add(Flatten(name = "FlattenedLayer"))

model.summary()


# In[ ]:


# Dense layer:

model.add(Dense(32))

model.add(Activation('relu'))

model.add(Dense(16))

model.add(Activation('relu'))

model.add(Dense(8))

model.add(Activation('sigmoid'))

model.add(Dense(1))

model.add(Activation('relu'))

model.summary()


# In[ ]:


# Compile model
model.compile(
              loss='binary_crossentropy',  # Metrics to be adopted by convergence-routine
              optimizer='rmsprop',         # Strategy for convergence?
              metrics=['accuracy'])        # Metrics, I am interested in

model.add(Activation('relu'))


# In[ ]:


# Dropout
model.add(Dropout(0.5))

model.summary()


# In[ ]:


# Compile model
model.compile(
              loss='binary_crossentropy',  # Metrics to be adopted by convergence-routine
              optimizer='rmsprop',         # Strategy for convergence?
              metrics=['accuracy'])        # Metrics, I am interested in

model.add(Activation('relu'))


# In[ ]:


#Image augmentation
def preprocess(img):
    return img


# In[ ]:


#Config1: Augmentation configuration for training samples
tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      # Normalize colour intensities in 0-1 range
                              shear_range=0.2,       # Shear varies from 0-0.2
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )


# In[ ]:


# Config2: Creating iterator from 'train_datagen'
train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       # Data folder of cats & dogs
                                               target_size=(img_width, img_height),  # Resize images
                                               batch_size=batch_size,  # Return images in batches
                                               class_mode='binary'   # Output labels will be 1D binary labels
                                                                     # [1,0,0,1]
                                                                     # If 'categorical' output labels will be
                                                                     # 2D OneHotEncoded: [[1,0],[0,1],[0,1],[1,0]]
                                                                     # If 'binary' use 'sigmoid' at output
                                                                     # If 'categorical' use softmax at output
                                                )


# In[ ]:


# Augmentation configuration for validation. Only rescaling of pixels
val_dtgen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


# validation data

validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )


# In[ ]:


# Model fitting

import time
start = time.time()
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        model.fit(x_batch, y_batch)
        batches += 1
        print ("Epoch: {0} , Batches: {1}".format(e,batches))
        if batches > 4:    # 200 * 10 = 2100 images
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

end = time.time()
(end - start)/60


# In[ ]:


# Model evaluation Using generator
result = model.evaluate(validation_generator,
                                  verbose = 1,
                                  steps = 4
                                  )


# In[ ]:


#  Result 'loss', 'accuracy'
result    


# In[ ]:





# In[ ]:




