#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning

# As computation power/resource as well as training time is the biggest concern in deep learning applications, this notebook will help people to gain an understand about how to leverage the benefits of transfer learning-that is how to load a pretrained model, fit this to own image datasets and use it in their own applications. This particular notebook used resnet50 model altough it is universal to any similar models.

# Specify Model

# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

my_new_model.layers[0].trainable = False


# Compile Model

# In[ ]:


my_new_model.compile(optimizer='sgd', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])


# Fit Model

# In[ ]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocess_input)

train_generator = data_generator.flow_from_directory(
                                        directory='../input/dogs-gone-sideways/images/train',
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
                                        directory='../input/dogs-gone-sideways/images/val',
                                        target_size=(image_size, image_size),
                                        class_mode='categorical')


fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=22,
                                       validation_data=validation_generator,
                                       validation_steps=1)


# **Please upvote if you like this or find this notebook useful, thanks.**
