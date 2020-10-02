#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will use the kuzushiji-mnist dataset to classify Japanese characters using a very simple Convolutional Neural Network based model.

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import optimizers
from keras.utils import to_categorical
from sklearn.utils import shuffle
import random

np.random.seed(1337)

import os
print(os.listdir("../input"))


# In[2]:


train_dir = "../input/"
train_img = np.load(train_dir + "kmnist-train-imgs.npz")['arr_0']
train_lbl = np.load(train_dir + "kmnist-train-labels.npz")['arr_0']
test_images = np.load(train_dir + "kmnist-test-imgs.npz")['arr_0']
test_labels = np.load(train_dir + "kmnist-test-labels.npz")['arr_0']
print(train_img.shape)
print(train_lbl.shape)
print(test_images.shape)
print(test_labels.shape)


# Lets create the validation data from training data. 

# In[3]:


train_img, train_lbl = shuffle(train_img, train_lbl, random_state = 0)
train_images = train_img[:50000]
train_labels = train_lbl[:50000]
validation_images = train_img[50000:]
validation_labels = train_lbl[50000:]


# In[4]:


train_images = train_images.reshape(50000, 28, 28, 1)
validation_images = validation_images.reshape(10000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)


# In[5]:


train_labels = to_categorical(train_labels)
validation_labels = to_categorical(validation_labels)
test_labels = to_categorical(test_labels)


# In[6]:


def plot_image(arr):
    plt.imshow(data, interpolation='nearest')
    plt.show()


# In[7]:


def create_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), activation = 'relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    return model


# In[8]:


model = create_model()
model.summary()


# In[9]:


opt = optimizers.RMSprop(lr = 1e-4)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['acc'])


# In[10]:


history = model.fit(train_images, train_labels, epochs = 20, batch_size = 10, 
                    validation_data = (validation_images, validation_labels), shuffle = True)


# In[11]:


model.save("kmnist_cnn.h5")


# In[12]:


hist = history.history
print(hist.keys())


# In[13]:


accuracy = hist['acc']
loss = hist['loss']
val_accuracy = hist['val_acc']
val_loss = hist['val_loss']


# In[14]:


len(accuracy)


# In[15]:


epochs = [i for i in range(1, 21)]


# In[16]:


plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)
plt.title("Accuracy vs Val Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()


# In[17]:


l, a = model.evaluate(test_images, test_labels)


# In[18]:


print(l, a)


# In[ ]:




