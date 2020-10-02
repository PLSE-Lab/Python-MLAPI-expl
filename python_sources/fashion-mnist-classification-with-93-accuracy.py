#!/usr/bin/env python
# coding: utf-8

# ### MNIST Image Classification on Fashion Clothing
# * In this project I am using MNIST dataset of fashion which is availabe in keras library.
# * ALTERNATE DATASET LINK: https://www.kaggle.com/zalando-research/fashionmnist

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow import keras
import tensorflow as tf


# In[ ]:


data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()


# In[ ]:


print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)


# In[ ]:


label_map = {0:"T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 
             4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
             9: "Ankle boot"}
print(label_map)


# ### Data Cleaning and Normalization

# In[ ]:


train_images = train_images/255.0
test_images = test_images/255.0


# ### Data Visualization

# In[ ]:


fig, axs = plt.subplots(1,4, figsize=(20, 10))
for ax, i in zip(axs, range(6)):
    ax.imshow(train_images[i])
    ax.grid(True)

plt.show()


# In[ ]:


unique, counts = np.unique(test_labels, return_counts=True)
df = pd.Series(zip(unique, counts))
print(df)
# All the types of clothers have same count present in the dataset ie. 1000


# ### Training data
# Training a neural network with one input layer, one hidden layer and one output layer for learning the digits in images.

# In[ ]:


from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler


# In[ ]:


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
print(train_images.shape, train_labels.shape)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
print(test_images.shape, test_labels.shape)


# In[ ]:


nclasses = train_labels.max() - train_labels.min() + 1
train_labels = to_categorical(train_labels, num_classes = nclasses)
print("Shape of train ylabels after encoding: ", train_labels.shape)


# In[ ]:


nclasses = test_labels.max() - test_labels.min() + 1
test_labels = to_categorical(test_labels, num_classes = nclasses)
print("Shape of test ylabels after encoding: ", test_labels.shape)


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.fit(train_images, train_labels, shuffle=True, batch_size=100, epochs=50, validation_split=0.2)


# ###  Evaluation of model over validation dataset

# In[ ]:


vc_loss, vc_accuracy = model.evaluate(test_images, test_labels)
print("\nLOSS: {}\nACCURACY: {}".format(vc_loss, vc_accuracy))


# ### I hope you like my approach.
# ### Please upvote to keep me going!!
