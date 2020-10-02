#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb

import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(101)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


# ## Load Dataset

# In[ ]:


# Load the data
train_digit = pd.read_csv("../input/train.csv")
test_digit = pd.read_csv("../input/test.csv")


# In[ ]:


test_digit.columns


# In[ ]:


train_digit.columns


# In[ ]:


X_train = train_digit.drop(labels = ["label"],axis = 1)
Y_train = train_digit['label']


# ## Count labels

# In[ ]:


sb.countplot(Y_train)


# ## Normalisation

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test_digit = test_digit / 255.0


# ## Reshaping

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)
test_digit = test_digit.values.reshape(-1,28,28,1)


# ## import Keras library through tensorflow

# In[ ]:


from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import utils
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import activations
from tensorflow.contrib.keras import callbacks
from tensorflow.contrib.keras import regularizers


# ## Label encoding

# In[ ]:


Y_train = utils.to_categorical(Y_train , num_classes=10)


# ## Split training and validation set

# In[ ]:


X_train ,X_test , Y_train , Y_test = train_test_split(X_train , Y_train , test_size = 0.2 , random_state = 101)


# ## Displaying images 

# In[ ]:


plt.imshow(X_train[5].reshape(28,28) ,cmap='gist_gray')


# ## Train the Model

# In[ ]:


model = models.Sequential()


# In[ ]:


model.add(layers.Conv2D(filters = 32,kernel_size = (4,4),padding = 'Same',activation = 'relu',strides = (1,1),input_shape = (28,28,1)))
model.add(layers.MaxPool2D(pool_size= (2,2) , strides = (1,1) , padding = 'Same'))

model.add(layers.Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same',activation = 'relu' ,strides = (1,1) ))
model.add(layers.MaxPool2D(pool_size= (2,2) , strides = (1,1) , padding = 'Same'))

model.add(layers.Conv2D(filters = 96,kernel_size = (3,3),padding = 'Same',activation = 'relu' ,strides = (1,1) ))
model.add(layers.MaxPool2D(pool_size= (2,2) , strides = (1,1) , padding = 'Same'))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation = 'relu' , kernel_regularizer= regularizers.l2(0.01) ))

model.add(layers.Dropout(0.25))
model.add(layers.Dense(10,activation = 'softmax'))


# ## Optimizer 

# In[ ]:


optimizer = optimizers.Adam(lr=0.001)


# ## Compile model

# In[ ]:


model.compile(optimizer= optimizer , loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# In[ ]:


model.fit_generator(datagen.flow(X_train,Y_train,batch_size=1000) , epochs=1 , validation_data=(X_test,Y_test) , steps_per_epoch=200)


# In[ ]:


# predict results
results = model.predict(test_digit)


# In[ ]:


# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

