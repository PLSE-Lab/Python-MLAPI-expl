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

from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Xtr=X_train
Ytr=y_train
Xte=X_test
Yte=y_test

import matplotlib.pyplot as plt
#plot the first image in the dataset to have an idea of the data we are dealing with

print(type(X_train))

import numpy as np

#Check the shape of the training images
print(X_train[0].shape)
print(X_test[0].shape)
X_train=X_train.reshape(60000,28,28,1)
X_test=X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical

#Perform One hot encoding for the train and test labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[0])
print(y_test[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D

##Initialise the model
model=Sequential()
##Add layers to the models
#Add Conv1 layer
model.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=(28,28,1)))
#Add pool1 layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Conv2 layer
model.add(Conv2D(64, kernel_size=3, activation='relu'))
#Add pool2 layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Conv2 layer
model.add(Conv2D(64, kernel_size=3, activation='relu'))
#Add pool2 layer
model.add(MaxPooling2D(pool_size=(2,2)))
#Flatten the  output from Conv 2 layer
model.add(Flatten())
#Add a desnse layer
model.add(Dense(128, activation='relu'))
#Finally add the output dense layer with softmax function activation
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)









# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


cls=model.predict_classes(X_test)
for i in range(30):
    plt.title("Predicted label:"+str(cls[i])+"\nGround truth label:"+str(Yte[i]))
    #plt.title("Ground truth label:"+str(Yte[i]))
    plt.imshow(Xte[i])
    plt.show()
   

