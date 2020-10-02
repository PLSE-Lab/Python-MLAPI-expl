#!/usr/bin/env python
# coding: utf-8

# Import Modules

# In[ ]:


import numpy as np
import pandas as pd
import cv2

import numpy as np
from numpy import genfromtxt

from keras import layers

from keras.layers import (Input, Dense, Activation, ZeroPadding2D,
BatchNormalization, Flatten, Conv2D, concatenate)

from keras.layers import (AveragePooling2D, MaxPooling2D, Dropout,
GlobalMaxPooling2D, GlobalAveragePooling2D)

from keras.models import Model, load_model
from keras import regularizers, optimizers

from keras.utils import to_categorical

import os

print(os.listdir('../input/fruits'))


# Loading the Training and Testing Data

# In[ ]:


def create_dictionary(directory):
    
    dict_labels = {}
    i = 0
    
    for x in os.listdir(directory):
        dict_labels[x] = i
        i = i+1
        
    return dict_labels

dict_labels = create_dictionary('../input/fruits/fruits-360_dataset/fruits-360/Training')


# In[ ]:


def create_train_data(directory):
    
    train_x = []
    train_y = []
    
    for x in os.listdir(directory):
        path = os.path.join(directory,x)
        
        for y in os.listdir(path):
            temp = os.path.join(path,y)
            img = cv2.resize(cv2.imread(temp),(50,50))
            train_x.append(img)
            train_y.append(dict_labels[x])
    
    return train_x,train_y

        
train_x,train_y = create_train_data('../input/fruits/fruits-360_dataset/fruits-360/Training')


# In[ ]:


train_x = np.array(train_x)
train_y = np.array(train_y)
train_y = np.reshape(train_y,[train_y.shape[0],1])
train_y = to_categorical(train_y)


# In[ ]:


test_x,test_y = create_train_data('../input/fruits/fruits-360_dataset/fruits-360/Test')
test_x = np.array(test_x)
test_y = np.array(test_y)
test_y = np.reshape(test_y,[test_y.shape[0],1])
test_y = to_categorical(test_y)


# In[ ]:


print("Train_x shape - ", train_x.shape)
print("Train_y.shape - ", train_y.shape)
print("")
print("Test_x.shape -  ", test_x.shape)
print("Test_y.shape -  ", test_y.shape)


# Defining and Training Model

# In[ ]:


def model(input_size):
    
    weight_decay = 0.0005
    
    x_input = Input(shape=(input_size,input_size,3))
    x = ZeroPadding2D((2,2))(x_input)
    
    x = Conv2D(16,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(103,activation='softmax')(x)
    
    model = Model(inputs=x_input,outputs=x,name='model')
    
    return model

model = model(50)

model.compile(loss='categorical_crossentropy',
             optimizer='Adam',
             metrics=['accuracy'])

model.fit(train_x,train_y,batch_size=32,epochs=18,validation_data=(test_x,test_y),shuffle=True)

model.save("Model-Fruits360")


# Testing Model

# In[ ]:


pred = model.evaluate(test_x,test_y)

print("Accuracy on Test Set = ", pred[1])


# In[ ]:


img = cv2.imread(''../input/testimage/test_image.jpg')


# In[ ]:




