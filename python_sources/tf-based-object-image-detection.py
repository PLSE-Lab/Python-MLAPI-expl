#!/usr/bin/env python
# coding: utf-8

# # Exercise Introduction
# 
# The cameraman who shot our deep learning videos mentioned a problem that we can solve with deep learning.  
# 
# He offers a service that scans photographs to store them digitally.  He uses a machine that quickly scans many photos. But depending on the orientation of the original photo, many images are digitized sideways.  He fixes these manually, looking at each photo to determine which ones to rotate.
# 
# In this exercise, you will build a model that distinguishes which photos are sideways and which are upright, so an app could automatically rotate each image if necessary.
# 
# If you were going to sell this service commercially, you might use a large dataset to train the model. But you'll have great success with even a small dataset.  You'll work with a small dataset of dog pictures, half of which are rotated sideways.
# 
# Specifying and compiling the model look the same as in the example you've seen. But you'll need to make some changes to fit the model.
# 
# **Run the following cell to set up automatic feedback.**

# In[ ]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_4 import *
import tensorflow as tf
print("Setup Complete")


# # 1) Specify the Model
# 
# Since this is your first time, we'll provide some starter code for you to modify. You will probably copy and modify code the first few times you work on your own projects.
# 
# There are some important parts left blank in the following code.

# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2 #number of output layers
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Indicate whether the first layer should be trained/changed or not.
my_new_model.layers[0].trainable = False

# Check your answer
step_1.check()


# # 2) Compile the Model
# 
# You now compile the model with the following line.  Run this cell.

# In[ ]:


my_new_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])


# That ran nearly instantaneously.  Deep learning models have a reputation for being computationally demanding.  Why did that run so quickly?
# 
# After thinking about this, check your answer by uncommenting the cell below.

# # 3) Review the Compile Step
# You provided three arguments in the compile step.  
# - optimizer
# - loss
# - metrics
# 
# Which arguments could affect the accuracy of the predictions that come out of the model?  After you have your answer, run the cell below to see the solution.

# # 4) Fit Model
# 
# **Your training data is in the directory `../input/dogs-gone-sideways/images/train`. The validation data is in `../input/dogs-gone-sideways/images/val`**. Use that information when setting up `train_generator` and `validation_generator`.
# 
# You have 220 images of training data and 217 of validation data.  For the training generator, we set a batch size of 10. Figure out the appropriate value of `steps_per_epoch` in your `fit_generator` call.

# In[ ]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(
                                        directory="../input/dogs-gone-sideways/images/train",
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
                                        directory="../input/dogs-gone-sideways/images/val",
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')

# fit_stats below saves some statistics describing how model fitting went
# the key role of the following line is how it changes my_new_model by fitting to data
fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=22,
                                       validation_data=validation_generator,
                                       validation_steps=1)


# Any suggestions are most welcome!!

# In[ ]:




