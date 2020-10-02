#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Cats and Dogs with CNN TensorFlow
# 
# This notebook is to recreate the code from [pythonprogramming.net](https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/) tutorials. The goal is to understand the code and how CNN models works in classifying and image of a cats or dogs using the Cats and Dogs [dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765).

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time


# ## Data

# In[ ]:


DATADIR = '../input/kagglecatsanddogs_3367a/PetImages'
CATEGORIES = ['Dog','Cat']


# ## Preprocess Dataset

# In[ ]:


#Iterate and convert datasets to an array.

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to dataset directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break


# In[ ]:


# Checking Data
print('Data Array:\n',img_array,'\n')
print('Data Shape:',img_array.shape)


# In[ ]:


# Resize the image
IMG_SIZE = 80
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


# In[ ]:


# Create dataset for training

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) # categories for dog(0) and cat(1)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
        
create_training_data()


# In[ ]:


print(len(training_data))


# In[ ]:


# Reshuffle Data
random.shuffle(training_data)

# Check the shuffled data
for sample in training_data [:10]:
    print(sample[1])


# In[ ]:


# Create list for training data
X = [] # feature dataset
y = [] # label dataset

for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert X to an array since you can't pass it to a neural network
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(X[:1])


# In[ ]:


# Save traing set using pickle

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()


# In[ ]:


# Open pickle file

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)
X[1]


# ## The Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


# In[ ]:


# Callback Name for Tensorboard
NAME = 'Cats-vs-Dogs-64x2-CNN-{}'.format(int(time.time()))


# In[ ]:


# Optimizing GPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# In[ ]:


# Load saved datasets

pickle_in = open('X.pickle','rb')
X = pickle.load(pickle_in)

pickle_in = open('y.pickle','rb')
y = pickle.load(pickle_in)


# In[ ]:


# Normalized Datasets
X = X/255.0


# In[ ]:


# Build Model

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #to convert 3D feature map to 1D
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir='./output/{}'.format(NAME))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])


# In[ ]:




