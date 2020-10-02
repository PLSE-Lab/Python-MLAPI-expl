#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kaggle-datasets-images/2243/3791/9384af51de8baa77f6320901f53bd26b/dataset-cover.png)

# # Fashion Class Classification
# 
# The [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist) dataset consists of 70.000 images,of witch 60.000 are training and 10.000 are testing samples. Dataset images are **28 x 28 pixel grayscale images**.
# 
# The target classes are:
# 0. T-shirt / top
# 1. Trouser
# 2. Pullover
# 3. Dress
# 4. Coat
# 5. Sandal
# 6. Shirt
# 7. Sneaker
# 8. Bag
# 9. Angle boot
# 
# Each image is 28 pixels in height and 28 pixels in width, totaling **784 pixels**. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, whith higher numbers meaning ligher. This pixel-value is an integer between **0 and 255**.

# ## Imports

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 

# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns # statistical data vizualization

# Sklearn
from sklearn.model_selection import  train_test_split

# Keras
import keras # open source Neural network library madke our life much easier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# Config
import warnings
warnings.simplefilter(action='ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Reading the data

# In[2]:


train_df = pd.read_csv('../input/fashion-mnist_train.csv')
test_df = pd.read_csv('../input/fashion-mnist_test.csv')

# shape of the data
print(train_df.shape)
print(test_df.shape)


# In[3]:


# printing the first rows of the train data
train_df.head()


# In[4]:


# printing the first rows of the test data
test_df.head()


# ### Exploratory Data Analysis

# In[5]:


# describing the train data
train_df.describe()


# In[6]:


# describing the test data
test_df.describe()


# In[7]:


# getting the info of the train data
train_df.info()


# In[8]:


# getting the info of the test data
test_df.info()


# In[9]:


# checking if the train dataset contains any NULL Values
null_counts = train_df.isna().sum()
null_counts = null_counts[null_counts > 0]

print(null_counts)


# In[10]:


# checking if the test dataset contains any NULL Values
null_counts = test_df.isna().sum()
null_counts = null_counts[null_counts > 0]

print(null_counts)


# ## Data Vizualization

# In[11]:


# Create training and testing arrays
training = np.array(train_df, dtype = 'int')
testing = np.array(test_df, dtype='int')


# In[12]:


class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


# In[13]:


# plot a random image
index = random.randint(1, len(test_df))

plt.imshow(training[index, 1:].reshape(28, 28))
label_index = training[index, 0]

print('Category: ', class_names[label_index])


# In[14]:


# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10, 10))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index
    label_index = training[index, 0]
    axes[i].imshow(training[index, 1:].reshape((28, 28)))
    axes[i].set_title(class_names[label_index], fontsize = 10)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.5)


# ## Feature Engineering

# In[15]:


X_train = training[:, 1:]/255
y_train = training[:, 0]

X_test = testing[:, 1:]/255
y_test = testing[:, 0]


# In[16]:


# splitting the data into training and validating sets
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2 )


# In[17]:


# unpack the tuple
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

print(X_train.shape)
print(X_test.shape)
print(X_validate.shape)


# ## Training the model

# In[18]:


cnn_model = Sequential()

cnn_model.add(Conv2D(64, 3, 3, input_shape = (28,28,1), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 32, activation = 'relu'))
cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))


# In[19]:


cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])


# In[ ]:


epochs = 50

history = cnn_model.fit(X_train,
                        y_train,
                        batch_size = 512,
                        nb_epoch = epochs,
                        verbose = 1,
                        validation_data = (X_validate, y_validate))


# In[ ]:


evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))


# In[ ]:


# get the predictions for the test data
predicted_classes = cnn_model.predict_classes(X_test)


# In[ ]:


L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() # 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("P: {:s}\n T: {:s}".format(class_names[predicted_classes[i]], class_names[y_test[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True, fmt='d')
# Sum the diagonal element to get the total true correct values


# In[ ]:


from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))

