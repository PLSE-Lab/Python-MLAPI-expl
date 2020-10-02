#!/usr/bin/env python
# coding: utf-8

# Transfer learning is a machine learning method, where a model developed for a task is reused as the starting point for a model on a second task.
# 
# In my very first Kaggle kernel, I make an attempt to see if the pre-trained VGG16 can be used to classify Kuzushiji Kmnist.

# **1.In this section, we load the relevant dependencies.**

# In[111]:


# Import relevant dependencies.

import numpy as np
import keras
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras.optimizers import Adam


# In[112]:


import os
print(os.listdir('../input'))

train_data = np.load('../input/kuzushiji/kmnist-train-imgs.npz')['arr_0']
test_data = np.load('../input/kuzushiji/kmnist-test-imgs.npz')['arr_0']
train_labels = np.load('../input/kuzushiji/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('../input/kuzushiji/kmnist-test-labels.npz')['arr_0']


# 2.Preparing the data:
# 
# This is how the rough VGG16 architecture looks like (as shown below).
# In this kernel, I will deviate slightly from the full VGG16 archiecture by using the convolutional layers, but not loading the last two fully connected layers which act as the classifier. 
# ![](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

# 

# In[115]:


##In this step, we flatten the individual arrays each data observation

train_data = np.array([np.array((train_data[i])) for i in range(len(train_data))])

train_data = train_data.flatten().reshape(60000, 784)

print("Shape of train_data: {}".format(train_data.shape))


test_data = np.array([np.array((test_data[i])) for i in range(len(test_data))])

test_data = test_data.flatten().reshape(10000, 784)

print("Shape of test_data: {}".format(test_data.shape))


# In the next three steps, I made use of the same steps listed in a previous Kaggle kernel's (Fashion Mnist) attempt to reshape the input to suit the VGG16 requirements.

# In[4]:


# Convert the images into 3 channels
train_data=np.dstack([train_data] * 3)
test_data=np.dstack([test_data]* 3)
train_data.shape,test_data.shape


# In[105]:


# Reshape images as per the tensor format required by tensorflow
train_data = train_data.reshape(-1, 28,28,3)
test_data= test_data.reshape (-1,28,28,3)
train_data.shape,test_data.shape


# In[106]:


# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
train_data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_data])
test_data = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_data])
#train_x = preprocess_input(x)
train_data.shape, test_data.shape


# 

# Other preprocessing steps

# In[107]:


# Normalise the data and change data type
train_data= train_data / 255.
test_data = test_data / 255.
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data.shape, test_data.shape


# In[108]:


# Converting Labels to one hot encoded format
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


# 3.Initialising the VGG16 neural net

# In[71]:


#Loading keras dependencies

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dense


# In[83]:


# Define the parameters for the VGG16 model
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 16


# In[84]:


#This is the setup for the VGG16 network
model_vgg16= VGG16(weights='../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                  include_top=False, 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)
                 )
model_vgg16.summary()


# In[85]:


# Freeze the layers except the last 4 layers
for layer in model_vgg16.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in model_vgg16.layers:
    print(layer, layer.trainable)


# In[94]:


model= Sequential()

model.add(model_vgg16)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()


# In[97]:


model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
  # optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])


# In[103]:


batch_size=128
epochs=12


# In[109]:


model.fit(train_data, train_labels_one_hot,
          batch_size=128,
          epochs=12,
          verbose=1,
          validation_data=(test_data, test_labels_one_hot))


# These are the results, which is only slightly better than that of simple convolution networks
# 
# 

# In[110]:


score = model.evaluate(test_data, test_labels_one_hot, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# To achieve state-of-the-art performance, other researchers have used an ensemble of VGG and Resnets.

# **References**
# 
# 1. A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning
# https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
# 
# 2. Kuzushiji-MNIST - Japanese Literature Alternative Dataset for Deep Learning Tasks
# https://towardsdatascience.com/kuzushiji-mnist-japanese-literature-alternative-dataset-for-deep-learning-tasks-d48ae3f5395b
# 
# 3. Fine-tune VGG16 Image Classifier with Keras | Part 1: Build
# https://www.youtube.com/watch?v=oDHpqu52soI
# 
# Fine-tune VGG16 Image Classifier with Keras | Part 2: Train
# https://www.youtube.com/watch?v=INaX55V1zpY
# 
# Fine-tune VGG16 Image Classifier with Keras | Part 3: Predict
# https://www.youtube.com/watch?v=HDom7mAxCdc&t=21s
# 
# 4. Keras Tutorial : Transfer Learning using pre-trained models
# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
# 
# 5. Various kaggle kernels:
# Classify Fashion_Mnist with VGG16
# https://www.kaggle.com/anandad/classify-fashion-mnist-with-vgg16
# EDA & Simple Keras CNN (K-MNIST)
# https://www.kaggle.com/xhlulu/eda-simple-keras-cnn-k-mnist/code
# 

# 

# 
