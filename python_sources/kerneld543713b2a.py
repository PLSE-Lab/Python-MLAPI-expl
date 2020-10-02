#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Firstly, let's import train and test file and look at their shape. The sample method will show the sample data of the provided dataframe.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
print("Shape of train.csv file ",train_df.shape)
test_df = pd.read_csv("../input/test.csv")
print("Shape of test.csv file", test_df.shape)
train_df.sample(5)
test_df.sample(5)


# In[ ]:


Y_train = train_df.iloc[:, 0].values
X_train = train_df.iloc[:, 1:].values
X_test = test_df.iloc[:, 0:].values
X_train = X_train.astype('float32')


# **Iloc** method returns the Series based on provided indices.  X_train stores the array of all values of pixels while Y_train stores the array of labels. And, we convert the type of X_train dataframe to float32. 

# In[ ]:


print("training data: {}".format(X_train.shape))
print("training labels {}".format(Y_train.shape))


# Did you notice the change in shape? The column of the X_train dataframe reduced by 1. ;) 

# In[ ]:


img_width=28
img_height=28
img_depth=1

X_train = X_train.reshape(len(X_train), img_width, img_height, img_depth)
X_test = X_test.reshape(len(X_test), img_width, img_height, img_depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)


# Reshaping of dataframe is done with reshape method. Division changes the pixel values ranging from 0 to 1.

# Okay, let's check our train data before doing anything to make sure we have valid input data.

# In[ ]:


plt.figure(figsize=(12,10))


# In[ ]:


for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Class {}".format(Y_train[i]))


# Now, we will train the model and predict the given digits.

# In[ ]:


Y_train = to_categorical(Y_train, num_classes=10)


# to_categorical converts a class vector (integers) to binary class matrix.

# So, it's time for stacking the layers into our model. Here, we will be using the Convolution2D. 

# In[ ]:


model = Sequential()
model.add(Convolution2D(32,(3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# Now, let's have a peak at our model.

# In[ ]:


model.summary()


# So, we use categorical_crossentropy for loss and adam optimizer for optimization.

# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Finally, we have come to the important step here. We will be fitting our model .

# In[ ]:


saved_path='cnn_model.h5'
model_checkpoint = ModelCheckpoint(filepath=saved_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.5, patience=6, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001, mode='auto')
callback_list = [model_checkpoint, early_stop, reduce_lr]

train_history = model.fit(X_train, Y_train, batch_size=32, 
                          epochs=15, verbose=1, callbacks=callback_list, validation_split=0.2)


# Now, let's do the fun part. **Prediction**.

# In[ ]:


model = load_model('cnn_model.h5')
labels = model.predict_classes(X_test)


# Let's check if our results are valid.

# In[ ]:


plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("predicted class {}".format(labels[i]))


# Seems okay, doesn't it?

# So, now we will submit our result through csv file.
# First step is to create a dataframe and conversion of the dataframe to csv file is done.

# In[ ]:


submit_df = pd.DataFrame({'ImageId': list(range(1, 28001)), 'Label': labels})


# In[ ]:


submit_df.to_csv('submit.csv', index=False)

