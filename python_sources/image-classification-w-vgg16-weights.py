#!/usr/bin/env python
# coding: utf-8

# In this notebook, we are going to use pretrained weights of VGG 16 and then add a new output layer with the required number of classes. In order to use the pretrained weights, you need add a new dataset containing the weights. Go to Data tab and click on 'Add Data Source'. Then search for 'Keras Pretrained Model' dataset which contains weights of different architectures like VGG16, Inception, Resnet50, Xception.
# 
# The dataset contains 3 directories: Training, Validation and Testing. Each directory contains sub-directories with images of different fruits.

# In[ ]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


# In[ ]:


# loading the directories 
training_dir = '../input/fruits/fruits-360_dataset/fruits-360/Training/'
validation_dir = '../input/fruits/fruits-360_dataset/fruits-360/Test/'
test_dir = '../input/fruits/fruits-360_dataset/fruits-360/test-multiple_fruits/'


# In[ ]:


# useful for getting number of files
image_files = glob(training_dir + '/*/*.jp*g')
valid_image_files = glob(validation_dir + '/*/*.jp*g')


# In[ ]:


# getting the number of classes i.e. type of fruits
folders = glob(training_dir + '/*')
num_classes = len(folders)
print ('Total Classes = ' + str(num_classes))


# In[ ]:


# this will copy the pretrained weights to our kernel
get_ipython().system('mkdir ~/.keras')
get_ipython().system('mkdir ~/.keras/models')
get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')


# In[ ]:


# importing the libraries
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
#from keras.preprocessing import image

IMAGE_SIZE = [64, 64]  # we will keep the image size as (64,64). You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
#x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


# Image Augmentation

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')


# We have not used any X_train, y_train, X_test, y_test or generated any labels for our classes. It is because we are using the flow_from_directory function of ImageDataGenerator. This function takes a directory as an input and assign the class indices to its sub directories. For this function to work, each subdirectory must contain a single class object only.
# Another function of ImageDataGenerator is 'flow'. With this function, we need to provide X_train, y_train. If you use X_train, y_train, X_test, y_test, don't forget to normalize X_train and X_test and use on hot encoding for y_train and y_test.

# In[ ]:


# The labels are stored in class_indices in dictionary form. 
# checking the labels
training_generator.class_indices


# In[ ]:


training_images = 37836
validation_images = 12709

history = model.fit_generator(training_generator,
                   steps_per_epoch = 10000,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results. 
                   epochs = 1,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = 3000)  # this should be equal to total number of images in validation set.


# In[ ]:


print ('Training Accuracy = ' + str(history.history['acc']))
print ('Validation Accuracy = ' + str(history.history['val_acc']))

