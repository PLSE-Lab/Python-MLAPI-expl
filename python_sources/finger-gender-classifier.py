#!/usr/bin/env python
# coding: utf-8

# # Intro 
# ##### welcome to my playground :)
# - In this kernel I am tring to make finger gender classifier by the use of fingerprints 
# - And that what i got so far, just a beginner, so any help or suggestion will be nice .. thank you :) :* 
# - i have problem with the Memory any suggestions ?? its almost full

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import os
import cv2
import random

import matplotlib.pyplot as plt


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split


# # labels
# 
# - so the labes for the dataset is not the folder name, instead its the img name
# - therefore I searched and get this function from:
#     * https://github.com/kairess/fingerprint_recognition
#     * https://www.kaggle.com/kairess/fingerprint-recognition
# - it return an numpy array with the subject ID, the person gender, left of right hand, and the finger gender
# 
# - Note:
#     - Training data labels are like this: 101__M_Right_ring_finger_Zcut
#     - Testing data labels are like this: 101__M_Right_ring_finger 
#     - so the split is different 
#     
# 

# In[ ]:


def extract_label(img_path,train = True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')
    
    if train:
        gender, lr, finger, _, _ = etc.split('_')
    else:
        gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
        
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)


# # Loading Data 

# In[ ]:


img_size = 96

def loading_data(path,train):
    print("loading data from: ",path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img),train)
            data.append([label[3], img_resize ])
        except Exception as e:
            pass
    data
    return data


# In[ ]:


Real_path = "../input/socofing/SOCOFing/Real"
Easy_path = "../input/socofing/SOCOFing/Altered/Altered-Easy"
Medium_path = "../input/socofing/SOCOFing/Altered/Altered-Medium"
Hard_path = "../input/socofing/SOCOFing/Altered/Altered-Hard"


Easy_data = loading_data(Easy_path, train = True)
Medium_data = loading_data(Medium_path, train = True)
Hard_data = loading_data(Hard_path, train = True)
test = loading_data(Real_path, train = False)

data = np.concatenate([Easy_data, Medium_data, Hard_data], axis=0)

del Easy_data, Medium_data, Hard_data


# # Preparing Data
# 
# * Create X as an array of pixels in img 
# * Reshape X
# * normalize X
# 
# * create y as the finger gender only and change it to onehot form 
# 
# repeat for test
# 
# * finally split the data into train, val

# In[ ]:


X, y = [], []

for label, feature in data:
    X.append(feature)
    y.append(label)
    
del data

X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0

y = to_categorical(y, num_classes = 5)


# In[ ]:


X_test, y_test = [], []

for label, feature in test:
    X_test.append(feature)
    y_test.append(label)
    
del test    
X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)
X_test = X_test / 255.0

y_test = to_categorical(y_test, num_classes = 5)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


print("full data:  ",X.shape)
print("Train:      ",X_train.shape)
print("Validation: ",X_val.shape)
print("Test:       ",X_test.shape)


# # Model
# * its from Yassine Ghouzam kernel on the NMIST Compition:
# * https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

# In[ ]:


epochs = 15
batch_size = 86


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (96,96,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(5, activation = "softmax"))

model.summary()


# In[ ]:


epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
model_path = './Model.h5'


model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer
callbacks = [
    EarlyStopping(monitor='val_acc', patience=20, mode='max', verbose=1),
    ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


# Without data augmentation i obtained an accuracy of 0.98114
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, y_val), verbose = 1, callbacks= callbacks)


# # Conclusion

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss,  label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

score = model.evaluate([X_test], [y_test], verbose=0)
print("Score: ",score[1]*100)

plt.show()

