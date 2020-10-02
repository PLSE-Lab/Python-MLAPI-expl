#!/usr/bin/env python
# coding: utf-8

# In[31]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dataset/dataset/"))

# Any results you write to the current directory are saved as output.


# ### set a string variable 'wd' for working directory to '../input/dataset/dataset'
# 

# In[32]:


wd = "../input/dataset/dataset/"


# ## Convolutional Neural Network
# 
# ### Part 1 - Building the CNN
# 
# Importing the Keras libraries and packages:
# Sequential, Conv2D, MaxPooling2D, Flatten and Dense

# In[33]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# # Initialize the CNN
# 

# In[34]:


model = Sequential()


# ### Step 1 - Convolution
# add a convolution layer with 32 units 3x3 shape. The input shape for the images is 64x64x3 and the activation layer 'relu'

# In[35]:


model.add(Conv2D(32,kernel_size=(3,3),input_shape=(64,64,3), activation='relu'))


# ### Step 2 - Pooling
# add a pooling layer for Max Pooling with a pool size of 2 by 2

# In[36]:


model.add(MaxPooling2D(pool_size=(2,2)))


# Adding a second convolutional layer which should be similar to the first one except one thing

# In[37]:


model.add(Conv2D(32,kernel_size=(3,3),input_shape=(64,64,3), activation='relu'))


# ### Step 3 - Flattening
# add a Flatten layer

# In[38]:


model.add(Flatten())


# ### Step 4 - Full connection

# Add two Dense layers with 128 and 1 units respectively. The first layer should cut off the negative values, whereas the second layer should return the classes in form of probabilities. Could you guess which activation functions are these?

# In[39]:


model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# ### Compiling the CNN
# compile the network with 'adam' optimizer, binary_crossentropy as a loss and accuracy as metrics

# In[40]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ### Augment the images. The code is given

# In[41]:


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(wd+'/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(wd+'/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# ### Fit the classifier
# ### Perform the training

# In[ ]:





# In[42]:


model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000, use_multiprocessing=True)


# ### Make a single prediction. The code is given

# In[50]:


#Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/dataset/dataset/sample/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
   prediction = 'dog'
else:
   prediction = 'cat'


# In[51]:


prediction


# In[ ]:




