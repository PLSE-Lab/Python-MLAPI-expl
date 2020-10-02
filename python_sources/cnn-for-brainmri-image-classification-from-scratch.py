#!/usr/bin/env python
# coding: utf-8

# ![](https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2018/05/CNN-Tensorflow-01.jpg)

# Hi, Let's Start from scratch 
# 
# First You have to **install** **tensorflow and keras**
# 
# to install tensorflow use :pip install tensorflow
# and to install keras use :pip install --upgrade keras
# 
# **Note**:- you may be need cuDNN tools for tensorflow it totally depends on purpose.
# 
# If you are using Anaconda then
# 1.open Anaconda prompt and give following commands
# 2.conda install tensorflow
# 3.conda install keras
# 
# **Note**:- installation differs from person to person you can use yours steps.

# Lets start with importing libraries

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# If You get similor output then tensorflow installed correctly. otherwise try to reinstall it.

# Now,start with Building CNN model

# In[ ]:


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# If you don't know the steps of CNN Google it.
# 
# Lets Pass the image dataset and automatically generation train and test dataset.

# In[ ]:


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/brain-mri-images-for-brain-tumor-detection/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/brain-mri-images-for-brain-tumor-detection/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch =300,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 60)


# 

# For more accuracy you can increase epoch size.Here i take 5 because it take to much time to compile.
# Then simply pass any image to the classifier but the image should be in same form as we have created the train dataset.
# 
# For this use Following Commented code 

# In[ ]:


# Part 3 - Making new predictions

# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction_image.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#    prediction = 'Have Tumer'
# else:
#    prediction = 'No Tumer'


# Try it on your laptop :-)
