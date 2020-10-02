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


# **Imports**
# 
# Import needed libraries and tools

# In[40]:


from sklearn.model_selection import train_test_split

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout
from keras.preprocessing.image import ImageDataGenerator


# **Data Preparation**
# 
# Some boilerplate code taken from Kaggle's tutorials on the Deep Learning track. What this does is setting up some variables that are going to be used later (img_rows/cols, num_classes) and defining a function to prepare the data from a raw csv file (read with pandas). More precisely, this function separates the labels into the y variable and the images into the tensor x, which is normalized by dividing for the maximum possible grey-scale value before being returned.

# In[46]:


img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_size = 30000
train_file = '../input/train.csv'
test_file = '../input/test.csv'
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)

random_seed = 5
x, x_val, y, y_val = train_test_split(x, y, test_size = 0.2, random_state=random_seed)


# **Data Augmentation**
# 
# Snippet taken from [kernel](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6). Using a data generator means that later on we'll have to use the fit_generator method on our model.

# In[47]:


datagen = ImageDataGenerator(
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


datagen.fit(x)


# **Building the model**
# 
# Next is the creation of the model with three convulutional layers, two with kernel size 2 and the last 3. These are purely arbitrary, and so is the number of filters used at each convolutional layer (the first argument in the Conv2D builder) and the activation function used for the layer.
# 
# A Flatten layer follows, and finally two dense, or fully-connected, layers to implement the softmax classifier. These latter are a typical "ending" for ConvNets as they allow the final classifying nodes to consider all of the features learned at the last convolutional layer before returning a "confidence" score for each class.

# In[48]:


model = Sequential()

model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(img_rows,img_cols,1)))

model.add(Conv2D(32, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# **Fit the model**
# 
# After having compiled the model, where you can set the type of loss and optimizer that you wish (and also the metrics you want it to show at each step of training), the last step of training is fitting the model to the training data.
# 
# In this method you should specify the size of your batch, that is the number of samples that will be propagated through the network at each training iteration (the loss is accumulated and backprop is only run after every batch); the number of epochs, which is the number of times you want your model to "go over" your training data; and finally, the validation split, that is the portion of your training dataset you wish to use for validation.

# In[50]:


#model.fit(x,y, batch_size=64, epochs=20, validation_split=0.2) #if we didn't use any data augmentation

batch_size = 64
model.fit_generator(datagen.flow(x,y, batch_size=batch_size),
                    validation_data= (x_val, y_val),
                    epochs = 20,
                    steps_per_epoch=x.shape[0] // batch_size)


# **Test Data Preparation**
# 
# This could have been done at the start together with the training data processing but it looked more clear when separate. We are doing the very same thing we have done with the traning data, except here we don't have any labels for our images to store in a variable y.  

# In[ ]:


test_data = pd.read_csv(test_file)
num_images = test_data.shape[0]
test_data_asarray = test_data.values[:,:]
test_data_shaped_array = test_data_asarray.reshape(num_images, img_rows, img_cols, 1)
t = test_data_shaped_array / 255


# **Make predictions**
# 
# The *predictions* variable will store an array of array of likelihoods for each class (number). The probability of an image being the *i-th* number is going to be stored at the *i-th* position in the array. We can't use directly this representation for the problem's solution, we only want to know the one number that has the greatest likelihood of being the number represented in the image.

# In[ ]:


predictions = model.predict(t)


# **Format predictions and create submission file**
# 
# In order to find the most likely classification for a given image, all we need to do is take the index of the maximum probability for each prediction. For an numpy array this is as easy as calling the argmax function on the nparray. Let's do this for all the predictions in the predictions 2-d array. Finally, we need to convert the predictions in the format requested by the challenge statement and convert the DataFrame to a .csv file ready for submission.

# In[ ]:


predictions_readable = [prediction.argmax() for prediction in predictions]
submission = pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
              "Label":predictions_readable})
submission.to_csv('submission.csv',index=False,header=True)

