#!/usr/bin/env python
# coding: utf-8

# **[Deep Learning Course Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# ---
# 

# # Exercise Introduction
# We will return to the automatic rotation problem you worked on in the previous exercise. But we'll add data augmentation to improve your model.
# 
# The model specification and compilation steps don't change when you start using data augmentation. The code you've already worked with for specifying and compiling a model is in the cell below.  Run it so you'll be ready to work on data augmentation.

# In[1]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_5 import *
print("Setup Complete")


# # 1) Fit the Model Using Data Augmentation
# 
# Here is some code to set up some ImageDataGenerators. Run it, and then answer the questions below about it.

# In[2]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224

# Specify the values for all arguments to data_generator_with_aug.
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True,
                                              width_shift_range = 0.1,
                                              height_shift_range = 0.1)
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)


# Why do we need both a generator with augmentation and a generator without augmentation? After thinking about it, check out the solution below.

# In[4]:


q_1.solution()


# # 2) Choosing Augmentation Types
# ImageDataGenerator offers many types of data augmentation. For example, one argument is `rotation_range`. This rotates each image by a random amount that can be up to whatever value you specify.
# 
# Would it be sensible to use automatic rotation for this problem?  Why or why not?

# In[5]:


q_2.solution()


# # 3) Code
# Fill in the missing pieces in the following code. We've supplied some boilerplate. You need to think about what ImageDataGenerator is used for each data source.

# In[6]:


# Specify which type of ImageDataGenerator above is to load in training data
train_generator = data_generator_with_aug.flow_from_directory(
        directory = '../input/dogs-gone-sideways/images/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

# Specify which type of ImageDataGenerator above is to load in validation data
validation_generator = data_generator_no_aug.flow_from_directory(
        directory = '../input/dogs-gone-sideways/images/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator, # if you don't know what argument goes first, try the hint
        epochs = 3,
        steps_per_epoch=19,
        validation_data=validation_generator)

q_3.check()


# In[7]:


q_3.hint()
q_3.solution()


# # 4) Did Data Augmentation Help?
# How could you test whether data augmentation improved your model accuracy?

# In[8]:


q_4.solution()


# # Keep Going
# You are ready for **[a deeper understanding of deep learning](https://www.kaggle.com/dansbecker/a-deeper-understanding-of-deep-learning/)**.
# 

# ---
# **[Deep Learning Course Home Page](https://www.kaggle.com/learn/deep-learning)**
# 
# 
