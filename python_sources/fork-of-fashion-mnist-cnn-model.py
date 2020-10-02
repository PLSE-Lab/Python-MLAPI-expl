#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# RESOURCE ==========
# * (CNN DETAILED EXPLAINATION) https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
# * https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb
# * https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l04c01_image_classification_with_cnns.ipynb#scrollTo=gut8A_7rCaW6

# In[ ]:


get_ipython().system('pip install tensorflow-datasets')

# Import TensorFlow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

#enable eager execution for image.numpy() method
tf.enable_eager_execution()
tf.executing_eagerly()


# Load the dataset and fetch its training and testing data

# In[ ]:


dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


# In[ ]:


train_dataset


# Since the class names are not included with the dataset, store them here to use later when plotting the images:

# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


# You get details of DATA from METADATA variable when the mnist is loaded

# In[ ]:


num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))


# We will now normalize the Pixel data (28x28 are there) . This will help correct any skew in data

# In[ ]:


print(metadata.supervised_keys)

#Normalize function is getting Images, Label from the mnist metadata.supervised_keys
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets. It is similar to df.apply()
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()


# In[ ]:


tf.executing_eagerly()


# Refer https://github.com/tensorflow/tensorflow/issues/27519 if you are unable to run below code
# 
# 
# The below code is taking 1 image as a 28x28 pixel value and then plotting it unsing plty

# In[ ]:


# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
  break
image = image.numpy().reshape((28,28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)

#Note the reason why we created class_names[] is because the label is a number in Dataset
plt.xlabel(label)
plt.colorbar()
plt.grid(False)
plt.show()


# Assemble the CNN Model architecture with the no. of Nuerions and the activation function to be used!!!!

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)
])


# Compile the NN with a Loss Function and a Gradient Descent function and also metrics 

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 1) Naive training
# 
# NOTE---
# ValueError: Error when checking input: expected flatten_2_input to have 4 dimensions, but got array with shape (28, 28, 1)
# 
# The above error comes when you do - model_1 = model.fit(train_dataset, epochs=5)
# 
# To avoid this make use of BATCH_SIZE
# 
# So .... the 4 dimentions are - 
# a)BATCH_SIZE
# b)28
# c)28
# d)1
# 

# In[ ]:


BATCH_SIZE = 32
model_1 = model.fit(train_dataset.batch(BATCH_SIZE), epochs=5)


# In[ ]:


model.save('Fashion_Classifier_CNN_Model.h5') 


# Also try out what google did in thier colab!!! 
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb#scrollTo=o_Dp8971McQ1

# In[ ]:




