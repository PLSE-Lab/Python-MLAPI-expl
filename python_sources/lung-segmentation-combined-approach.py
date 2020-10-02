#!/usr/bin/env python
# coding: utf-8

# Credits: Code and technique combined from the following 3 sources
# * https://www.kaggle.com/toregil/a-lung-u-net-in-keras
# * https://www.kaggle.com/irrwitz/cnn-with-keras
# * https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/run/755223
# 
# This model combines preprocessing of the images with a CNN Sequential model and data augmentation by applying random rotation to the images and masks. The accuracy on the test set is 98.5%

# In[ ]:


import tensorflow as tf
import numpy as np
np.random.seed(123)
import os
from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.io import imread
from tensorflow.python.framework import ops
import math

import pandas as pd
import keras.backend as K

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation,                          Flatten, Convolution2D, MaxPooling2D,                          BatchNormalization, UpSampling2D
from keras.utils import np_utils
from scipy import ndimage
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border


# In[ ]:


K.set_image_dim_ordering('th')
# helper function to load the image and downsample it by 4
jimread = lambda x: np.expand_dims(imread(x)[::4, ::4],0)
#load images and masks
data_path = os.path.join('..', 'input')

all_images = glob(os.path.join(data_path, '2d_images', '*.tif'))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]
print(len(all_masks), 'matching files found')


# * Read the inages and masks, which are now at 1/4th resolution, into a stack.
# * Scale all mask values to 0-1.
# * Preprocess the images by first creating true/false map of the Houndsfield Unit values. 
# * clear_border gets rid of the body and CT machine components.
# * By multipying the original images by the cleared true/false images, we get the a rough lung segmentation.

# In[ ]:


images = np.stack([jimread(i) for i in all_images], 0)
masks = np.stack([jimread(i) for i in all_masks], 0) / 255.0

#Preprocess
preImages = np.stack([imread(i)[::4, ::4] for i in all_images], 0)
for idx,image in enumerate(preImages):
    binary = image < -400
    cleared = clear_border(binary)
    preImages[idx] = preImages[idx] * cleared
    
#plt.imshow(preImages[0])
images = np.expand_dims(preImages, 1)
print(images.shape)


# In[ ]:


X_train, X_test, y_train,  y_test = train_test_split(images, masks, test_size=0.1)
print('Training input is', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is', X_test.shape)


# The generator expands our image set by giving the images a random rotation. By using the same random seed, the images and masks will still correlate.

# In[ ]:


SEED=42
#Data augmentation
def image_augmentation_generator(xtrain, ytrain, batch_size):
    data_generator = ImageDataGenerator(
            rotation_range=45).flow(xtrain, xtrain, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
           rotation_range=45).flow(ytrain, ytrain, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


# Lets take a look at a preprocessed image and the corresponding mask.

# In[ ]:


first_image = images[0]
first_mask = masks[0]
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(first_image[0])
ax2.imshow(first_mask[0])


# In[ ]:


# Create a deep nn
model = Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3, 3), 
                        activation='relu', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(Convolution2D(filters=64, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dense(64, activation='relu'))
model.add(Convolution2D(filters=1, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(UpSampling2D(size=(2,2)))

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy','mse'])
#print(model.summary())


# In[ ]:


history = model.fit_generator(image_augmentation_generator(X_train, y_train, 16),
                           steps_per_epoch = 60,
                           validation_data = (X_test, y_test),
                           epochs=10, verbose=1)


# In[ ]:


#Graphs of the accuracy and loss
fig, ax = plt.subplots(1,2)
ax[0].plot(history.history['acc'], 'b')
ax[0].set_title('Accuracy')
ax[1].plot(history.history['loss'], 'r')
ax[1].set_title('Loss')


# 

# In[ ]:


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize = (20, 8))
ax1.imshow(X_test[0,0])
ax2.imshow(y_test[0,0])
ax3.imshow(model.predict(X_test)[0,0], cmap="Greys")
ax4.imshow(X_test[1,0])
ax5.imshow(y_test[1,0])
ax6.imshow(model.predict(X_test)[1,0], cmap="Greys")


# In[ ]:


scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


# serialize model to JSON
model_json = model.to_json()
modelFileName = "kaggle_model_"
with open(modelFileName + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelFileName + ".h5")
print("Saved model to disk")


# In[ ]:




