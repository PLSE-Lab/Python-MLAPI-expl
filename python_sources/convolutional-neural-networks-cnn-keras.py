#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks (CNN) - Keras
# 
# Created by: Sangwook Cheon
# 
# Date: Dec 31, 2018
# Updated: Jun 27, 2019
# 
# This is step-by-step guide to Convolutional Neural Network (CNN), which I created for reference. I added some useful notes along the way to clarify things. This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of CNN.
# 
# ## Content:
# ### Explanation
# 1. Convolution operation
# 2. ReLU
# 3. Max Pooling
# 4. Flattening
# 5. Full-connection
# Softmax & Cross entropy
# 
# ### IMPLEMENTATION
# 1. Data preprocessing
# 2. Build the Keras model
# 3. Compile and fit the model
# 4. Make predictions and determine accuracy
# 
# --------- 

# Below are key concepts used in Convolutional Neural Networks.
# 
# # Convolution Operation
# The aim of convolution operation is to reduce the size of an image, by using feature detectors that keep only the specific patterns within the image. Stride is the number of pixels with which we slide the detector. If it is one, we are moving it one pixel each time and recording the value (adding up all the multiplied values). Many feature detectors are used, and the algorithm finds out what is the optimal way to filter images. 3 x 3 feature detector is commonly used, but other sizes can be used.
# 
# # ReLU
# After feature detectors are applied upon images, ReLU is used to increase non-linearity within images. 
# 
# # Max Pooling
# Take a 2 x 2 box on the top left corner (starting here), and record the maximum number within the box. Slide it to the right with the stride of 2 (commonly used), and move onto the next row if completed. Repaet this step until all the pixels are evaluated. Aim of max pooling is to keep all the important features even if images have spatial or textual distortions, and also reduce the size which prevents overfitting. So, after applying convolution operation to images, than pooling is applied.
# 
# Other pooling techniques are also available such as Mean Pooling, which takes the average of pixels within the box.
# 
# # Flattening
# Flatten the matrix into a long vector which will be the input to the artificial neural network
# 
# # Full Connection
# Implement full Artificial Neural Network model to optimize weights.
# 
# # Softmax & Cross entropy
# Softmax function brings all predicted values to be between 0 and 1, and make them add up to 1. It also comes hand-in-hand with cross-entropy method. 
# 
# Just seeing how many wrong predictions the classifier made is not enough to evaluate the performance of ANNs. Instead, Cross Entropy should be used to measure how good the model is, as there can be two models that produce same results while one produced better percentages than the other. For classificaion, Cross Entropy should be used, and for regression, Mean Squared Error should be used. 
# 
# ---------
# 
# We will create a dog vs cat classifier. To be able to work with keras library, we need proper structure of images. There should be two folders: Test set and Train set. And, in each folder, cat images and dog images should be placed in two separate folders. In this way, keras will understand how to work with them.
# 

# ## 1. Build the CNN model

# In[ ]:


# import libraries
# we are not importing libraries used for csv files, as keras deals with all of these

from keras.models import Sequential
from keras.layers import Convolution2D #images are two dimensional. Videos are three dimenstional with time.
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the classifier CNN
classifier = Sequential() #Please note that there is another way to build a mode: Functional API.

#applying convolution operation --> build the convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#32, 3, 3 --> 32 filters with 3 x 3 for each filter. 
#start with 32 filters, and then create more layers with 64, 128, 256, etc
#expected format of the images.
# 256, 256, 3 --> 3 color channels (RGB), 256 x 256 pixels. But when using CPU, 3, 64, 64 --> due to computational limitation


# In[ ]:


#Max Pooling --> create a pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))
# 2 x 2 size --> commonly used to keep much information.

#Flattening --> creating a long vector.
classifier.add(Flatten()) #no parameters needed.

#classic ANN - full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#common practice: number of hidden nodes between the number of input nodes and output nodes, and choose powers of 2
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


# ## 2. Compile the model

# In[ ]:


classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


# ## 3. Fit the model on images, image preprocessing
# Data augmentation prevents overfitting, by generating more samples of the images through flipping, rotating, distorting, etc. Keras has built-in Image Augmentation function. To learn more about this function, refer to this [guide](https://keras.io/preprocessing/image/). 

# In[ ]:


#Data augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('../input/training_set', 
                                                    target_size = (64, 64), 
                                                    batch_size = 32,
                                                   class_mode = 'binary')
test_set = test_datagen.flow_from_directory('../input/test_set',
                                                target_size = (64, 64),
                                                 batch_size = 32, 
                                                 class_mode = 'binary')

classifier.fit_generator(training_set, 
                         samples_per_epoch = 8005, 
                        nb_epoch = 2, 
                        validation_data = test_set, 
                        nb_val_samples = 2025)


# ## 4. Improving the model
# There are two possible ways of reducing variane, which is making the model fit more to the train set.
# * Add more convolutional layers 
#     * This will allow more features to be detected prior to fitting to ANN. Make sure to not include input_dim, and include MaxPooling step. Flattening should be at the end of all the layers.
# * Add more fully-connected layer (hidden layers)
#     * Catches more complex behaviors
#     
# Thank you for reading this kernel. If you found this helpful, please upvote the kernel or put a short comment below. 
