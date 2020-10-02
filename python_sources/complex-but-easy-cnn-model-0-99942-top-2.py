#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# **SimonXu**
# 
# Undergrad in ICL
# 
# Electronic Engineering

# To make things clear, the 0.99942 accuracy was achieved by using **ALL** of the MNIST dataset. I made use of the 'mnist-in-csv' dataset by combining the training and eval files into one csv file containing 70000 samples.
# 
# You may regard this as **CHEATING**, but the network itself achieves 0.99571 when using the dataset provided by kaggle, which is quite an acceptable result.
# 
# P.S. If you used the model but didn't achieve the results that I mentioned above, then you simply were unlucky. There certainly is some inevitable randomness in ML.
# 
# P.P.S. I would like to thank Yassine Ghouzam for creating a great kernel, most of the code is based on his kernel (Because I mainly use tensorflow instead of keras)
# 
# P.P.P.S. I will share a version using tensorflow in the soon future.

# ## Preperation Stage
# 
# 
# **Importing libraries**
# 
# First, we need to import all the necessary libraries.

# In[ ]:


import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# **Reading data**
# 
# Now we import the raw data stored in the csv files by calling 'pd.read_csv()'
# 
# Y_train stands for the training labels, while X_train stands for training images.
# 
# We do not want our training images containing the labels, so we call 'drop()' to drop the labels column.
# 
# 

# In[ ]:


train = pd.read_csv('../input/train.csv') 
test = pd.read_csv('../input/test.csv')

Y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1)


# **Data processing**
# 
# The images were stored in gray-scale values (0 to 255), so we need to normalize it by dividing the values by 255 to simplify calculations.

# In[ ]:


X_train = X_train / 255.0
test = test / 255.0


# We also need to reshape the data into a form that convolutions can be performed easily.
# 
# Before reshaping, the images were stored in 1D arrays. After reshaping, the images are now in shape of 3D matrices (28 x 28 x 1). 
# 
# The 3D matrices have depth of 1 because the images are in black and white. If the images were colored, the matrices would have a depth of 3, standing for each of the three channels (RGB).
# 
# P.S. Note that we've indicated -1 for batch size, which specifies that this dimension should be dynamically computed based on the number of input values.

# In[ ]:


X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)


# We now transform the labels into categorical data, so we can use 'categorical_crossentropy' for our loss.
# 
# The labels are now stored as one-hot vectors that have lengths of 10.
# For example, a label of 7 would now be stored as [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], and 3 would become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# In[ ]:


Y_train = to_categorical(Y_train, num_classes = 10)


# **Preparing the evaluation set**
# 
# To makes things simple, we use the function 'train_test_split'.
# 
# 'test_size' determines how much of the training set will be used for evaluation. Here I used 0.2, which means 20% of the training set will be used for evaluating the model.
# 
# 'random_seed' is something that is purely up to you. Note that random_state doesn't only take integers as an input, you can find out more by googling 'train_test_split()'.

# In[ ]:


random_seed = 10
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = random_seed)


# ## Defining the model
# 
# Here is where the fun really begins, you get to create and play with your own model. 
# 
# If you don't quite know how convolutional networks work, you can access: https://cs231n.github.io/convolutional-networks/ to learn about it. You can also find out more about great models like VGGNet, ResNet in the link.
# 
# Convolutional networks mostly follow the path: **Convolutional layers** >> **Pooling layers** >> **Fully Connected layers**
# 
# Convolutional layers are used to dig up features that are useful for classification. Typically, more filters lead to better performance at a cost of longer training time.
# 
# It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. In most cases, max pooling gives the best result, but average pooling shows good results as well in some specific circumstances. For a max pooling layer with kernel size of 2, stride of 2 (this is the most common configuration) downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations, but the depth remains constant.
# 
# Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. FC layers often make up most of the learnable parameters in CNN models. Since two layers of FC layers can approximate most non-linear functions, adding a third one can improve perfomance, but if the number of layers go further than that, one can hardly find improvements in performance.
# 
# In order to overcome overfitting, dropout is applied to both maxpooling layers and FC layers. Note that the dropout probabilities are different for the two types of layers. For pooling layers, dropout probablities around 0.2 works well. As for FC layers, dropout probablities around 0.5 works best.
# 
# Batch normalization is also added between FC layers and activation to speed up convergence. Most researches claim that BN works better when placed before activations.
# 
# The final FC layer is the output layer, the 10 neurons stand for the 10 classes (0 to 9). This layer generates the raw values for our prediction. Softmax activation is added to derive the probablities from the raw values.
# 
# Note that convolutional layers take up most of the GPU memory used, while FC layers contribute to most of the training time. So reduce the numbers of layers if neccessary.

# In[ ]:


model = Sequential()

model.add(Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu',
    input_shape = (28, 28, 1)))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 128,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 256,
    kernel_size = (3,3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

model.add(Conv2D(
    filters = 512,
    kernel_size = (3, 3),
    padding = 'same',
    activation = 'relu'))

#This 1 x 1 conv layer is added to increase depth and add some non-linearity to the model. 1 x 1 conv layers were introduced in Google Inception Net,
#which shown great results.
model.add(Conv2D(
    filters = 1024,
    kernel_size = (1, 1),
    padding = 'same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))


# **Compiling the model**
# 
# For the optimizer, I chose Adam because Adam is superior over most other optimizers in most cases, and it's rather easy to tweak with. What you only need to care about is the learning rate, Adam itself takes care of the rest. Learning rates too high or too low can make convergence too slow. undesirable or even impossible.
# 
# Batch size and epochs are also hyperparameters that you can tweak with. Epochs stand for the number of batches that will be fed to the model to train. I set it to 200 because the model contains quite a lot of learnable parameters, making the convergence rather slow. Batch size defines how many training samples are contained in one batch, popular choices are 32, 64, 128, etc. Larger batch sizes lead to better performance, but at a cost.

# In[ ]:


optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08) 

#compile the model
#for loss, I chose categorical crossentropy
#it should be noted that categorical crossentropy can only be used for categorical data
#that's why the labels were transformed in the code above
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

epochs = 1 #you really need to change this, this is just for convinience
batch_size = 128


# **Data augmentation**
# 
# Data augmentation has shown to be a great way of tackling overfitting and improving performance. It basically produces new images and adds them to the training samples based on rules that you set.  It works particularly well when training samples are scarce.

# In[ ]:


datagen = ImageDataGenerator(
    rotation_range = 10, #randomly rotates the image by 10 degrees
    zoom_range = 0.1, #randomly zooms the image by 10%
    width_shift_range = 0.1, #randomly shifts the image horizontally by 10% of the width
    height_shift_range = 0.1 #randomly shifts the image vertically by 10% of the height
)


# **Training and logging**
# 
# This is where we train the model and generate logs. Try setting 'verbose' to 1 to use the super cool logging style.

# In[ ]:


history = model.fit_generator(datagen.flow(x = X_train, y = Y_train, batch_size = batch_size),
    epochs = epochs, validation_data = (X_val, Y_val),
    verbose = 2, steps_per_epoch = X_train.shape[0] // batch_size)


# ## Prediction & Output
# 
# By calling 'predict()', we get the class probabilities. The class with the highest probablity should be selected to be the label so we use 'np.argmax'.
# 
# Finally, we store our results in the way kaggle requires and submit it manually.
# 
# Enjoy the sense of achievement when the score comes out, lol.

# In[ ]:


results = model.predict(test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name = "Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)

submission.to_csv("cnn_mnist_datagen.csv", index = False)


# **This is the first tutorial I have ever posted, so I still have a lot to learn. **
# 
# **If you enjoyed this kernel, that's great news. If you have any questions or I have made any mistakes, please don't hesitate to leave it down below.**
