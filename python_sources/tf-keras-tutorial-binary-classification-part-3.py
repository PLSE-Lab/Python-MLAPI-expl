#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Keras Tutorial - Binary Classification (Part 3)
# 
# **What is Keras?** Keras is a wrapper that allows you to implement Deep Neural Network without getting into intrinsic details of the Network. It can use Tensorflow or Theano as backend. This tutorial series will cover Keras from beginner to intermediate level.
# 
# <p style="color:red">IF YOU HAVEN'T GONE THROUGH THE PART 1 and 2 OF THIS TUTORIAL, IT'S RECOMMENDED FOR YOU TO GO THROUGH THAT FIRST.</p>
# [LINK TO PART 1](https://www.kaggle.com/akashkr/tf-keras-tutorial-neural-network-part-1)<br>
# [LINK TO PART 2](https://www.kaggle.com/akashkr/tf-keras-tutorial-cnn-part-2)<br>
# 
# In this part we will cover:
# * Using Image Data Generator
# * Using Augmentation to increase variety in Dataset
# * Plotting Accuracy and Loss for Training and Validation

# ## Importing Libraries

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import os


# ## Path
# We define path to training and test data. The path to the folder must contain directories of classes.

# In[ ]:


training_images_path = '../input/dogs-cats-images/dataset/training_set'
validation_images_path = '../input/dogs-cats-images/dataset/test_set'


# ## Image Data Generator
# We define **Image Data Generator** which takes image data from folder and structures it to feed into our model.<br>
# There are two steps for initialising a data generator. First is setting the preprocessing parameters and then defining the path and the size of the data.
# 
# ### ImageDataGenerator
# `ImageDataGenerator` class generates batches of tensor image data with real-time data augmentation.
# 
# > * **rescale** Factor to be multiplied to each pixel value
# * **rotation_range** Images sampled after rotating to a maximum of + and - rotation_range
# * **width_shift_range** Images sampled after randomly shifting to left or right at given percentage
# * **height_shift_range** Images sampled after randomly shifting to above or below at given percentage
# * **shear_range** Images sampled after rotation on X, Y and Z axis
# * **zoom_range** Zooms image IN and OUT at given percentage
# * **horizontal_flip** Flip horizontally and samples
# * **fill_mode** Fill mode of blank space after transformation
# 
# ### ImageDataGenerator.flow_from_directory()
# `ImageDataGenerator.flow_from_directory()` function to generate data using path.
# > * **path** Image directory
# * **target_size** Dimension to scale each image before training
# * **batch_size** Size into which image data is divided to feed in model
# * **class_mode** Target variable type

# In[ ]:


# Defining training image generator with all the augmentation sample parameters
# This ensures correct classification for different image in validation
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Defining validation image generator
# We don't pass augmentation parameters as we model haven't seen this data earlier
validation_generator = ImageDataGenerator(rescale=1/255)

# Loading training data from path
train_generator = train_datagen.flow_from_directory(
    training_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)

# Loading validation data from path
validation_generator = validation_generator.flow_from_directory(
    validation_images_path,
    target_size=(150, 150),
    batch_size=40,
    class_mode='binary'
)


# ## Modelling
# To see the details of layers and models, see the PART 1 and PART 2 of this tutorial.

# In[ ]:


model = tf.keras.models.Sequential([
    # First Convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Second Convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Third Convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten
    tf.keras.layers.Flatten(),
    # Dense layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)


# In[ ]:


model.summary()


# #### Training model
# Instead of training and validation data, we pass their generator.

# In[ ]:


history = model.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=20,
    verbose=1,
    validation_data=validation_generator
)


# ## Loss and Accuracy
# Lets plot loss and accuracy across epochs that is stored in the history object while training.
# 
# > Observing the graph, you can see that the training and validation accuracy rises till 10 epochs then the curves flatten.
# Similarly you can see that the training and validation loss decreases till 10 epochs and the curve flattens.
# Note that if the training accuracy increases and validation decreases, that is an indication of **Overfitting**.

# In[ ]:


# Getting the accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting the accuracy
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

# Plotting the loss
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# **IN THE NEXT TUTORIAL WE WILL SEE HOW TO LOAD A PRE-TRAINED MODEL AND USE IT FOR MULTICLASS CLASSIFICATION.**
# 
# > # PART 4 [Using Pretrained Models and Multiclass Classification](https://www.kaggle.com/akashkr/tf-keras-tutorial-pretrained-models-part-4)
