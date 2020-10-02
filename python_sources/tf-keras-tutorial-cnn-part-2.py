#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Keras Tutorial - Convolutional Neural Network (Part 2)
# 
# **What is Keras?** Keras is a wrapper that allows you to implement Deep Neural Network without getting into intrinsic details of the Network. It can use *Tensorflow* or *Theano* as backend. This tutorial series will cover Keras from beginner to intermediate level.
# 
# <font color=red>IF YOU HAVEN'T GONE THROUGH THE PART 1 OF THIS TUTORIAL, IT'S RECOMMENDED FOR YOU TO GO THROUGH THAT FIRST.</font><br>
# [LINK TO PART 1](https://www.kaggle.com/akashkr/tf-keras-tutorial-neural-network-part-1)
# 
# In this part we will cover:
# * Basics of Convolutional Neural Network
# * Image Data basic preprocessing for CNN
# * Training Convolutional Neural Network
# * Visualizing data in intermediate layers
# 
# # WHAT IS CONVOLUTIONAL NEURAL NETWORK?
# Convolutional Neural Network comprises of Convolutional layers. A Convolutional layer is the layers that focus on neighbouring/local properties of an image rather the complete input image.
# 
# **CONVOLUTION LAYER**<br>
# A n*n 2D matrix with some predefined values is called a **kernel**. The kernel moves step by step on the entire image. Each value is multiplied with the corresponding inplace value on image and summed to get the output value. This picture will make it clearer.
# 
# <img src='https://miro.medium.com/max/1400/1*TwMfGALfE0naUC8pLOK4Vg.jpeg' width=500>
# 
# Note that the size of the input reduces after convolution layer because it can't process values at the edges. For a 3x3 kernel size convolution, a NxN image converts to N-1xN-1 size. 
# 
# **POOLING**<br>
# Pooling layers are used to reduce the dimension of the image. There are many types of pooling layers in CNN like *Max Pooling, Average Pooling, Global Pooling* etc. The most common is Max Pooling.
# * **Max Pooling**<br>
# In max pooling the pixel with maximum value is the output on a pool size of nxn. This image will make it more clear.
# 
# <img src='https://media.geeksforgeeks.org/wp-content/uploads/20190721025744/Screenshot-2019-07-21-at-2.57.13-AM.png' width=500>
# 
# You would have seen **Dense** and **Flatten** layer in the previous part. So we will be using that here. THIS WAS AN OVERVIEW, JUST TO GIVE YOU SOME FEELING OF WHAT IS GOING INSIDE. NOW LET'S DIVE INTO THE CODE TO GET MORE INTUITIONS.

# # Importing libraries

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset
# 
# Here we are loading mnist Dataset which is preloaded in tensorflow. <br>
# 
# >```mnist = tf.keras.datasets.mnist```<br>
# This returns the dataset object. Similarly there are 6 more datasets preloaded in keras.
# 
# >Calling the `load_data` function on this object returns splitted train and test data in form of (features, target).

# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 


# # Overview of Dataset
# 
# The dataset contains images, each image of 28x28px. There are 6000 images in training data and 1000 images in test data.<br>
# >The shape (6000, 28, 28) represents **6000** images each of dimension **28x28**.<br>
# The shape **(6000, )** represents (6000, 1) shape i.e. 6000 labels, each for one image.

# In[ ]:


print(f'Shape of the training data: {x_train.shape}')
print(f'Shape of the training target: {y_train.shape}')
print(f'Shape of the test data: {x_test.shape}')
print(f'Shape of the test target: {y_test.shape}')


# In[ ]:


print(y_train)


# In[ ]:


# Let's plot the first image in the training data and look at it's corresponding target (y) variable.
plt.imshow(x_train[0], cmap='gray')
print(f'Target variable is {y_train[0]}')


# # Preprocessing
# 
# >```python
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# ```
# 
# 
# In these lines of code we reshape the data to feed in the Model. You can see the data reshaped to `(6000, 28, 28, 1)`. Let's see what are these numbers. `6000` is the number of images, `28, 28` is the shape of image and `1` is the number of channels in the image. **Grayscale images have 1 channel. When using colored images this can be replace by 3 i.e. (6000, 28, 28, 3) when using colored images**
# 
# The next step is Normalizing i.e. scaling the pixels to 0-1 from 0-255.

# In[ ]:


# Reshaping the data
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Normalizing
x_train = x_train/255
x_test = x_test/255


# # Modelling
# 
# There are two types of models in Tensorflow:
#  - **Sequential**
#  - **Graphical**
# 
# ## Models
# `tf.keras.model.Sequential()` 
# lets you create a linear stack of layers providing a Sequential netural network.<br>
# `tf.model()`
# allows you to create arbitarary graph of layers as long as there is no cycle.
# 
# ## Convolution Layer
# `tf.keras.layers.Conv2D()` Convolution layer takes the following argument
# > * **filter** Number of different types of convolutions used. Initially they are set to some predefined convolution and slowly trained to find better features in the image.
# * **kernel_size** An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
# * **activation** activation function
# * **input_shape** Size of each input to the convolution.
# 
# ## Max Pooling
# `tf.keras.layers.MaxPooling2D()` Max Pooling layer to reduce the size of the input.
# > * **pool_size** Dimension of pooling kernel
# 
# ## Flatten Layer
# `tf.keras.layers.Flatten()` flattens the input.<br>
# For input of `(batch_size, height, width)` the output converts to `(batch_size, height*width)`
# 
# ## Dense Layer
# `tf.keras.layers.Dense()` Normal dense layer of Neural Network where each node is connected to each node in the next layer. <br>
# >The two arguments passes below in dense layer are *units* and *activation* (activation function).<br>
# * **units** corresponds to the number of nodes in the layer<br>
# * **activation** is an element-wise activation function.
#     * **relu**: This activation function converts every negative value to 0 and positive remains the same
#     * **softmax**: This function takes the element with max value, converts it into 1 and rest to 0.
# 
# In the below example we've 3 dense layers with 128, 64 and 10 nodes respectively. The layer with 10 nodes is supposed to be the output layer. Since we just need single value as 1 (preferably the maximum value), therefore we apply *softmax* activation function to the final/output layer.
# 
# ## Compiling model
# `model.compile()` Sets up the optimiser, loss and metrics configuration.
# > * **optimizer**: updates the parameter of the Neural Network.
# * **loss**: Measures the error in our model.
# * **metrics**: Used to judge the model. The difference between metrics and loss is that metrics in not used to evaluate the model while training, whereas loss evaluates the model error while training and helps optimizer reduce the error.

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ## Model details
# 
# Let's look at details of the model.
# * The 3x3 pixel kernel reduces the image size by 1 pixel on every size because it can't process the pixels at the edges and the 64 types of filter outputs 64 layers of matrix converting to a output of **(26, 26, 64)**.
# * The Max Pooling of pool size 2x2 reduces the image by half. Output **(13, 13, 64)**.
# * The next layers are again a convolution and max pooling with the net output of shape **(5, 5, 64)**.
# * The flatten layer unrolls the input to a single dimension array i.e. 5 x 5 x 64 = 1600
# * The number of nodes in the next layer is fixed by us to 128
# * Final output layer contains 10 nodes for 10 classes.

# In[ ]:


model.summary()


# ## Training
# 
# ```model.fit``` trains the model.
# > * **x_train**: Training data/features
# * **y_train**: Target
# * **epochs**: Number of times the entire dataset is fed in the model
# 
# While training you can see the loss and accuracy calculated on the training data itself. *The number of epochs are a kind of try and test metrics. It depends on a number of factors like size of data and complexity of classification, etc.* You will slowly get a feeling of how to estimate number of epochs required for a particular model and dataset.
# 
# ## Validation
# 
# We've verified our model's accuracy for training data which comes out to be 97% approx. This accuracy is calculated on the same data on which the model is trained. Now let's see the accuracy when our model gets new data on which it is not trained. Generally it should be slightly lower than the training accuracy.<br>
# > This is done to avoid overfitting. **Overfitting** is the case when the model gives *high accuracy on training data but low accuracy on test data*. Whenever you see a hugh difference in the Train accuracy and Validation accuracy, it is an example of overfitting. 
# Overfitting can be caused due to a number of reasons and have several ways to avoid and fix. We will discuss that later in the course.<br>
# *Note that we expect the validation accuracy to be slightly lower than training accuracy but not much lower.*

# In[ ]:


# Training
model.fit(x_train, y_train, epochs=5)

# Validation
test_loss = model.evaluate(x_test, y_test)


# # Visualising image stages in layers

# In[ ]:


f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1

# Constructing a custom model with input as our model input and output as all the layers' output
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

# Taking 3 random images and visualizing its one convolution
for x in range(0,4):
  f1 = activation_model.predict(x_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)

  f2 = activation_model.predict(x_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
    
  f3 = activation_model.predict(x_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)


# **IN THE NEXT TUTORIAL WE WILL SEE HOW TO PROCESS REAL LIFE DATASET AND SOME TECHNIQUES TO AUTO LABEL TRAIN AND VALIDATION DATA FROM FOLDER**
# 
# > # PART 3 [Binary Classification](https://www.kaggle.com/akashkr/tf-keras-tutorial-binary-classification-part-3)
