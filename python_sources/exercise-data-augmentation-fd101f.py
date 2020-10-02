#!/usr/bin/env python
# coding: utf-8

# # Exercise Introduction
# We will return to the automatic rotation problem you worked on in the previous exercise.
# 
# We also supply much of the code you've already worked with.  Fork this notebook and take on the data augmentation step (step 2 below).

# # 1) Specify and Compile the Model
# This works the same way as in the code you've previously worked on. So you receive a complete version of it here.  Run this cell to specify and compile the model

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# # 2) Fit the Model Using Data Augmentation
# 
# Fill in the blanks, and uncomment those lines of code.  After doing that, you can run this cell and you should get a model that achieves about 90% accuracy.  By using data augmentation, you cut the error rate in half.

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224

# Specify the values for all arguments to data_generator_with_aug. Then uncomment those lines
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                              horizontal_flip = True,
                                              width_shift_range = 0.2,
                                              height_shift_range = 0.2)
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)


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
         # specify where model gets training data
         train_generator,
         epochs = 3,
         steps_per_epoch=19,
         validation_data=validation_generator) # specify where model gets validation data


# # Keep Going
# You are ready for [a deeper understanding of deep learning](https://www.kaggle.com/dansbecker/a-deeper-understanding-of-deep-learning/).
