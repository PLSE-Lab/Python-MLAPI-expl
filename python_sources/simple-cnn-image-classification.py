#!/usr/bin/env python
# coding: utf-8

# the dataset has images of forests,seas,buildings,glaciers,mountains and streets. this notebook attempts to predict them as a particular class/category using CNN layers and keras image preprocessing

# In[ ]:


#import basic libraries
import numpy as np
import pandas as pd


# In[ ]:


import keras


# **basic understanding about layers used **
# 
# Sequential:Sequential layer simply acts an inital layer, for the model layers to move in the particular sequence
# 
# Convolution2D: it is the first layer of CNN. Convolution is a mathematical operation to merge two sets of information. In our case ,the input image and the feature detector ,are both merged to create a feature map. 
# 
# MaxPooling2D:maxpooling creates a maxpooling layer, the only argument is the window size.i have used the 3*3 window size.
# 
# Dense: layer used to add the fully connected layers to neural networks
# 
# Flatten: After the convolution and pooling layers we flatten their output to feed into the neural networks
# 
# 

# In[ ]:


#import all the essential packages from keras , required for CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten




# In[ ]:



classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))  # 32 is the number of feature detectors, and 3*3 window

classifier.add(MaxPooling2D(pool_size=(2,2)))    #size of the sub table ,min is considered = 2*2 

classifier.add(Flatten())       # flattens the sub table into a linear to be able to fed into training


# In[ ]:


classifier.add(Dense(128,activation = 'relu'))      # first and input layer
classifier.add(Dense(6,activation = 'softmax'))    #output layer, 6 is the no of output categories 


# In[ ]:


#compile the model

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#sparse categorical bcz the output has multiple catgories


# In[ ]:


#image data preprocessing 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2, #shaering transformation
        zoom_range=0.2,       # zoom in required
        horizontal_flip=True) #if the images data needs to be horizontally flipped, applicable for real world images

test_datagen = ImageDataGenerator(rescale=1./255) #rescale the image if necessary (RGB coefficients normalize)


# In[ ]:


#creates the train set

train_set = train_datagen.flow_from_directory("../input/intel-image-classification/seg_train/seg_train",
        target_size=(64,64), #size of the image in the model 
        batch_size=32,
        class_mode='sparse')


# In[ ]:


#creates the test set
test_set = test_datagen.flow_from_directory(
        '../input/intel-image-classification/seg_test/seg_test',
        target_size=(64,64),       #size of the image in the model
        batch_size=32,
        class_mode='binary')


# In[ ]:


#fit the model

classifier.fit_generator(train_set,
                    steps_per_epoch=14034,     #number of images
                    epochs=5,
                    validation_data=test_set,
                     validation_steps=3000)

