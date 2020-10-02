#!/usr/bin/env python
# coding: utf-8

# # Learning Outcome 3
# ### Digit recognition with the MNIST dataset
# 
# This notebook is split into 2 different approaches we used to obtain high scores. 
# Both approaches use a fully connected network. Where one approach uses a kFold to reduce bias and variance, the other uses data augmentation to enlargen the dataset for training.
# 
# _Disclaimer:_  
# We decided to split the different approaches because they were both very different, but interesting.
# After combining the two approaches we obtained the incredible score of:
# ![image.png](attachment:image.png)
# 
# _Annotated by:_
# > Jordi

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold, train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Used to track how long the model is training
import time


# # Indicator 3.1
# This competition has it's own train- and test set. Pandas is used to put the two .csv files in dataframes.
# After running the right cells (whether for Kaggle or for locally stored data), train data is stored in the variable 'train' and test data is stored in the variable 'test'
# 
# ## When using Kaggle
# When you are using Kaggle the next cells must be executed to load in the data.
# On Kaggle the input data files are available in the "../input/" directory.
# 
# _Annotated by:_
# > Jano  
# > Jordi

# In[ ]:


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# Read in the different datafiles
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")


# ## When using this notebook locally
# When running the notebook locally the next cells must be executed to load in the data.
# 

# In[ ]:


train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")


# ## Data preparation

# In[ ]:


# Set 'label' column as targets of trainset
Y_train = train["label"]

# Drop 'label' column from trainset, to only leave the features (aka the pixels)
X_train = train.drop(labels = ["label"],axis = 1)


# In[ ]:


# Print the distribution of the digits present in the trainset
print('Label   Count    Percentage')
for i in range(0,10):
    print("%d       %d     %.2f" % (i, Y_train.value_counts()[i], round(Y_train.value_counts(normalize=True)[i]*100, 2)))


# In[ ]:


# Divide values by 255 to get an input value between 0 and 1 for every pixel
X_train = X_train / 255.0
test = test / 255.0


# In[ ]:


# Creating copies to use later in the kFold approach
X_train_K = X_train.copy()
Y_train_K = Y_train.copy()
test_K = test.copy()


# ## Data augmentation approach

# In[ ]:


# reshaping the data to rank 4 so I can use Data Augmentation

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

X_train.shape


# In[ ]:


Y_train = keras.utils.to_categorical(Y_train, num_classes=10)


# # Indicator 3.1
# To validate if the neural network is actually learning, the train dataframe is split up in another train set and a validation set.
# 
# > ~Jano

# In[ ]:


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=42)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)


# # Indicator 3.2
# I tried adding layers and tried removing layers, but I got the best score by only using one layer.
# I kept increasing the density of my layer untill the model stopt improving. It stopt improving at 1024
# I tried different dropouts to make the neural network less vulnerable to overfitting. 0.5 improved my score the most
# > ~Jano

# In[ ]:


model_A = keras.models.Sequential()

# Flattening the data so I can fit the rank 4 data
model_A.add(Flatten())

# For some reason got the best score using only one layer
# Kept making the layer denser and it kept improving my score, it stopt improving arround 1024
model_A.add(keras.layers.Dense(1024, activation='relu', input_shape=(784,)))

# Add dropout to my layer to make it less vulnerable to overfitting
model_A.add(Dropout(0.5))

model_A.add(keras.layers.Dense(10, activation="softmax"))


# # Indicator 3.2
# I use the adam optimizer because it is good at getting out of the local minum.
# I use data augmentation to create more images based on the train set, so the neural network has more images to train itself on. I create more images by rotating the images, zooming the images in and out, shifting the images on the x axis and shifting the images on the y axis.
# > ~Jano

# In[ ]:


model_A.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Add the parameters for Data Augmentation
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

# Fit the Data Augmentation to my training set
datagen.fit(X_train)


# In[ ]:


X_train.shape


# In[ ]:


history = model_A.fit_generator(datagen.flow(X_train,Y_train, batch_size=32),
                              epochs = 55, validation_data = (X_val,Y_val))


# In[ ]:


predictions_A = model_A.predict_classes(test)
print(predictions_A)


# In[ ]:


my_submission_A = pd.DataFrame({'ImageId': list(range(1,len(predictions_A)+1)), 'label': predictions_A})

# you could use any filename. We choose submission here
my_submission_A.to_csv('submission_A.csv', index=False)


# # Indicator 3.1 - 3.2 - 3.3
# ## kFold approach
# 
# This approach uses a kFold to create train- and validationsets
# Using a kFold has 2 major benefits:  
#  -- The bias of the model is reduced, because more data can be used for fitting  
#  -- The variance of the model is reduced, because more data can be used for validation
# 
# In this approach a sequential deep learning model is used. This model, a so called fully connected neural network, consists of a linear stack of dense layers. Each neuron in a layer is connected to every neuron in the preceding and succesive layers.  
# The amount of neurons per layer is based on trying different amounts of neurons and the design described [here](https://en.wikipedia.org/wiki/MNIST_database).
# The optimizer used is the Adam optimizer. This choice was based on trying different optimizers and [this article](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).
# 
# _Annotated by:_
# > Jordi
# 

# In[ ]:


# Remove anything but the values of the X_train dataframe
X_train_K = X_train_K.values

print(X_train_K.shape)
print(Y_train_K.shape)


# In[ ]:


kf = KFold(n_splits = 10,
           shuffle=True)


# In[ ]:


# Create a copy of the labels to use for printing samples of the train and testsets
# Samples are printed to give a general idea of train and testsets of the different folds
Y_labels = Y_train_K
for train_idx, test_idx in kf.split(X_train_K):
    _train = plt.figure(figsize=(20,2))
    for i in range(1,11):
        ax = _train.add_subplot(1, 10, i)
        ax.imshow(X_train_K[train_idx[i-1]].reshape(28, 28))
        ax.set_xlabel(Y_labels[train_idx[i-1]])
    _train.suptitle('Trainsample of in total %d records' % len(train_idx), fontsize=14)
    plt.show()
    
    _test = plt.figure(figsize=(20,2))
    for i in range(1,11):
        ax = _test.add_subplot(1, 10, i)
        ax.imshow(X_train_K[test_idx[i-1]].reshape(28, 28))
        ax.set_xlabel(Y_labels[test_idx[i-1]])
    _test.suptitle('Testsample of in total %d records' % len(test_idx), fontsize=14)
    plt.show()


# In[ ]:


# Convert Y train values into a matrix with 10 columns, a column for each class
#   (Comparable to hot-encoding)
Y_train_K = keras.utils.to_categorical(Y_train_K, num_classes=10)


# In[ ]:


model_K = keras.models.Sequential()

model_K.add(keras.layers.Dense(784, activation='relu', input_shape=(784,)))
model_K.add(keras.layers.Dense(800, activation="relu"))
model_K.add(keras.layers.Dense(10, activation="softmax"))


# In[ ]:


# Configure the learning process
model_K.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy', 'mse']
)


# In[ ]:


# Summarize the model, this gives information about the amount of parameters (weights & biases)
model_K.summary()


# In[ ]:


# Keep track of the running time by storing the starttime
start_time = time.time()

# Fit the model for every fold in the kFold
for train_idx, test_idx in kf.split(X_train_K):
    model_K.fit(
        X_train_K[train_idx],
        Y_train_K[train_idx],
        batch_size=32,
        epochs=15,
        validation_data=(X_train_K[test_idx], Y_train_K[test_idx])
    )
    
# Calculate the runtime by substracting the starttime from the current time
runtime = time.time() - start_time
print("/n--- Runtime of %s seconds ---" % (runtime))


# In[ ]:


# Use the trained neural network to identify the digits in the testset
predictions_K = model_K.predict_classes(test.values)
print(predictions_K)


# In[ ]:


# Create a dataframe from the predictions, made by the neural network
my_submission_K = pd.DataFrame({'ImageId': list(range(1,len(predictions_K)+1)), 'label': predictions_K})

# Save the predictions in the file 'submission.csv'
my_submission_K.to_csv('submission_K.csv', index=False)


# # Final score of 0.98096
# Settings for final score:
# <ul>   
#     <li>Data split with KFold into 10 folds</li>
#     <li>Connected Neural Network with 2 hidden layers:</li>
#     <ul>
#         <li>784 neurons, relu activation</li>
#         <li>800 neurons, relu activation</li>
#     </ul>
#     <li>Settings for learning:</li>
#     <ul>
#         <li>Optimizer: Adam</li>
#         <li>Loss function: categorical_crossentropy</li>
#     </ul>
#     <li>Model fitted with:</li>
#     <ul>
#         <li>Batch size of 32</li>
#         <li>15 epochs</li>
#     </ul>
# </ul>
# 
# Running time of 849 seconds
# 
# Overview of scores:
# ![image.png](attachment:image.png)
# 

# In[ ]:




