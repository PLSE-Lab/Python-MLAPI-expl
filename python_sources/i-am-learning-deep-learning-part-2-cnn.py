#!/usr/bin/env python
# coding: utf-8

# # Content
# * Introduction
# * Import Library
# * Load Data
# * Sample Data Representation
# * Data Augmentation
# * Data Split Train And Test
# *  Create Convolutional Neural Network Model
# * Conclusion

# # INTRODUCTION
# * I wouldn't try my second deep learning experience.
# * I created and tested a convolutional neural network model using the Keras library.
# * I wish you good readings.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load Data
# * In this data there are 2062 sign language digits images.
# * At the beginning of tutorial we will use only sign 0 and 1 for simplicity. 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).
# * Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1).

# In[ ]:


# load data
X = np.load('../input/Sign-language-digits-dataset/X.npy')
Y = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64
print('X shape: ',X.shape)
print('Y shape: ',Y.shape)


# # Sample Data Representation
# * We are plotting our sample data.

# In[ ]:


# sample data representation
plt.subplot(1, 2, 1)
plt.imshow(X[700], cmap = 'gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(X[900], cmap = 'gray')
plt.axis('off')
plt.show()


# # Data Augmentation
# If we have larger and diverse data in machine learning, we can better train our models. In order to achieve this in image processing, we can diversify our data by rotating, moving, zooming in or out of the images.

# In[ ]:


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10
)


# # Data Split Train And Test
# * We divide 80% of our data into training and 20% test data.

# In[ ]:


# test and train splite
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
print('x_train shape: ',x_train.shape)
print('x_test shape: ',x_test.shape)
# axis representing grey-scale
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
print('x_train shape: ',x_train.shape)
print('x_test shape: ',x_test.shape)


# In[ ]:


# We fit the data generator with our training data.
datagen.fit(x_train)


# # Create Convolutional Neural Network Model
# * conv => max pool => dropout => conv => max pool => dropout => fully connected
# * Our deep learning model consists of two hidden layers.

# In[ ]:


# Create CNN model
model = Sequential()
# firs layer
model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
# second layer
model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.3))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


# Change the learning rate
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)


# In[ ]:


# model compile
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# * We will first train our model with real data.

# In[ ]:


history1 = model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, verbose=0)


# In[ ]:


print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score[0],score[1]))
plt.plot(history1.history['loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# * We are now training our model with the data we transform.

# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer = optimizer ,metrics=['accuracy'])
history2 = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),steps_per_epoch=64, epochs=10)
score = model.evaluate(x_test, y_test, verbose=0)


# In[ ]:


print('Loss: {:.4f}  Accuaracy: {:.4}%'.format(score[0],score[1]))
# Plot the loss and accuracy curves for training and validation 
plt.plot(history2.history['loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# * Let's visualize our model's predictions.

# In[ ]:


# confusion matrix
# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# # Conclusion
# * A notebook I've made to try what I've learned more about deep learning and try to better understand deep learning.
# * I'm new to programming. I'm even more new in data science, machine learning, deep learning and artificial intelligence. But I am working. And I'il be an artificial intelligence developer. Your comments are very important to me.
# * Thank you for reading my notebook. Waiting for your criticism.
