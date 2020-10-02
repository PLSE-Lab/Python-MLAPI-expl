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


# This is my first kernal and any feedback are really appreciate.
# 
# 

# In[ ]:


#First things first so import the packages and read the data using pandas as below

from keras.datasets import mnist
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.cm as cm
import numpy as np

# use Keras to import pre-shuffled MNIST database
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
data_train = pd.read_csv('../input/train.csv')
X_train = data_train.loc[:, data_train.columns != 'label']
Y_train = data_train['label']

X_test = pd.read_csv('../input/test.csv')




print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))


# Convert the data sets into matrix and reshape the inputs to 28 x 28 since each image is 768 pixels

# In[ ]:


X_test = X_test.as_matrix()
X_test = X_test.reshape(len(X_test),28,28)

X_train = X_train.as_matrix()
X_train = X_train.reshape(len(X_train),28,28)

# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# Visualize any one image and check how the pixels values are spread across the digit, The pixel value on the image will help us to identify the pattern.
# 

# In[ ]:


def visualize_input(img, ax):       
    ax.imshow(img, cmap='gray')    
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(X_test[2000], ax)
#visualize_input(X_train[250], ax)
#print(Y_train[0])


# Now lets check labels and corresponding value. We will convert the labels into binary format.

# In[ ]:


from keras.utils import np_utils

# print first ten (integer-valued) training labels
print('Integer-valued labels:')
print(Y_train[250])

# one-hot encode the labels
Y_train = np_utils.to_categorical(Y_train, 10)
#y_test = np_utils.to_categorical(y_test, 10)

# print first ten (one-hot) training labels
print('One-hot labels:')
print(Y_train[250])


# Create the Convolution Neural Network using Keras

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
X_train = X_train.reshape(len(X_train),28,28,1)


# define the model
model = Sequential()
#model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=16, kernel_size=2, strides=2,padding='same', activation='relu', input_shape=(28, 28,1)))
#model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=2, strides=2,padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=2, strides=2,padding='same', activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))
#model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
# summarize the model
model.summary()


# Compile the model with Adam optimizer

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])


# Now it's the fun part to execute the Model and share the model with TensorBoard. As I used Keras to build the model, I delared the TensorBoard callback and passed the callback in the model execution.

# In[ ]:


from keras.callbacks import ModelCheckpoint,TensorBoard 
import tensorflow as tf
# train the model
checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 
                               verbose=1, save_best_only=True)
#tf.summary.scalar('CheckPoint',checkpointer)
Board = TensorBoard(log_dir='../output/logs', histogram_freq=0, batch_size=1000, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

hist = model.fit(X_train, Y_train, batch_size=1000, epochs=5,
          validation_split=0.2, callbacks=[checkpointer,Board],
          verbose=1, shuffle=True)


# While the model is still in execution we can start the Tensor board with below command
# 
# Tensorboard --logdir='<<Your complete path the log directory' 
# 
# please note logdir should be complete path any relative paths will not be accepted and throws an error message
# 
# Once the TensorBoard recognize the path and it gives below path 
# http://yourmachinename:6006
# this path might work in few machine if your machine recognize if not open the chrome and paste below path to open
# http://localhost:6006
# 
# Here you go, you can see the grpah as below
# 
# ![image.png](attachment:image.png)

# Click on SCALARS tab, to see the metrics as below 
# 
# ![image1.png](attachment:image.png)
# 
# 

# Here it is , it is run time so you can see the metrics as your model keeps training with each epochs.
# ![image2.png](attachment:image.png)

# For more information about TensorBoard
# https://www.tensorflow.org/guide/summaries_and_tensorboard
# 
# Kera callback Tensor Board
# https://keras.io/callbacks/
# 
# Give me thumb up if you like it and leave any comments and feedback. 
# 
# Thanks
# 

# In[ ]:




