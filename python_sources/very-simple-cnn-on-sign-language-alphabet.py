#!/usr/bin/env python
# coding: utf-8

# ## About 
# This notebook contains a very fast and simple convolutional neural network (CNN) example in Python.
# 
# This work is part of a series called [Deep learning - very fast fundamental examples](https://www.kaggle.com/jamiemorales/deep-learning-very-fast-simple-examples)
# 
# The approach is designed to help grasp the applied artificial intelligence workflow in minutes. It is not an alternative to actually taking the time to learn. What it aims to do is help someone get started fast and gain intuitive understanding of the typical steps early on.

# ## Step 0 Understand the problem
# What we're trying to do here is classify sign language alphabets.

# ## Step 1: Set-up and understand data
# In this step, we layout the tools we will need to solve the problem identified in the previous step. We want to inspect our data sources and explore the data itself to gain an understanding of the data for preprocessing and modeling.

# In[ ]:


# Set-up libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer


# In[ ]:


# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load data
train_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

train_labels = np.array(train_images['label'].values)
train_images = np.array(train_images.drop('label', axis=1).values)

test_labels = np.array(test_images['label'].values)
test_images = np.array(test_images.drop('label', axis=1).values)


# In[ ]:


# Explore a few items
plt.figure(figsize=(12,12))
for i in range(1,21):
    plt.subplot(4,5,i)
    plt.imshow(train_images[i].reshape(28,28))


# ## Step 2: Prepare data and understand some more
# In this step, we perform the necessary transformations on the data so that the neural network would be able to understand it. Real-world datasets are complex and messy. For our purposes, most of the datasets we work on in this series require minimal preparation.

# In[ ]:


# Reshape and normalise data
train_images_number = train_images.shape[0]
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height*train_images_width

train_images = train_images / 255.0
train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)


test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

test_images = test_images / 255.0
test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)


# In[ ]:


# Transform labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)


# In[ ]:


# Explore data some more
print('Shape of training: ', train_images.shape)
print('Shape of training labels: ', train_images.shape)

print('Number of images: ', train_images_number)
print('Height of image: ', train_images_height)
print('Width of image: ', train_images_width)
print('Size of image: ', train_images_size)

print('\nShape of test: ', test_images.shape)
print('Shape of training labels: ', train_images.shape)

print('Number of images: ', test_images_number)
print('Height of image: ', test_images_height)
print('Width of image: ', test_images_width)
print('Size of image: ', test_images_size)


# ## Step 3: Build, train, and evaluate neural network
# First, we design the neural network, e.g., sequence of layers and activation functions. 
# 
# Second, we train the neural network, we iteratively make a guess, calculate how accurate that guess is, and enhance our guess. The first guess is initialised with random values. The goodness or badness of the guess is measured with the loss function. The next guess is generated and enhanced by the optimizer function.
# 
# Lastly, use the neural network on previously unseen data and evaluate the results.

# In[ ]:


# Build and train neural network
model = tf.keras.Sequential([
    keras.layers.Conv2D(64, (8,8), padding='same', activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(24, activation='softmax')
])

# Compile neural network
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']
             )

# Train the neural network
model.fit(train_images, train_labels, epochs=5)


# In[ ]:


# Apply the neural network
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))


# There's much to be improved but we'll leave it here for now and take this further in future examples.

# ## Learn more
# If you found this example interesting, you may also want to check out:
# 
# * [Deep learning - very fast fundamental examples](https://www.kaggle.com/jamiemorales/deep-learning-very-fast-simple-examples)
# * [Machine learning in minutes - very fast fundamental examples in Python](https://www.kaggle.com/jamiemorales/machine-learning-in-minutes-very-fast-examples)
# * [List of machine learning methods & datasets](https://www.kaggle.com/jamiemorales/list-of-machine-learning-methods-datasets)
# 
# Thanks for reading. Don't forget to upvote.
