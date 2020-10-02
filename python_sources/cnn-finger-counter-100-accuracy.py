#!/usr/bin/env python
# coding: utf-8

# ## Importing the images

# In[ ]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from skimage import io, transform

import os, glob

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[ ]:


train_images = glob.glob("../input/fingers/train/*.png")
test_images = glob.glob("../input/fingers/test/*.png")


# In[ ]:


X_train = []
X_test = []
y_train = []
y_test = []
    
for img in train_images:
    img_read = io.imread(img)
    # Most images are already of size (128,128) but it is always better to ensure they all are
    img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_train.append(img_read)
    # The info about the number of fingers and the fact that this is a right or left hand is in two characters of the path
    y_train.append(img[-6:-4])
    
for img in test_images:
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (128,128), mode = 'constant')
    X_test.append(img_read)
    y_test.append(img[-6:-4])


# In[ ]:


y_train[:5]


# Those are the labels, the number indicates the number of fingers and the letter indicates the orientation.
# 
# Now let's look at some images.

# In[ ]:


io.imshow(X_train[0])
plt.show()
io.imshow(X_train[1])
plt.show()
io.imshow(X_train[2])
plt.show()
io.imshow(X_train[3])
plt.show()


# In[ ]:


X_train = np.array(X_train)
X_test = np.array(X_test)


# In[ ]:


print(X_train.shape,X_test.shape)


# In[ ]:


X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


# We need to use one hot encoding on the labels to use a CNN.

# In[ ]:


label_to_int={
    '0R' : 0,
    '1R' : 1,
    '2R' : 2,
    '3R' : 3,
    '4R' : 4,
    '5R' : 5,
    '0L' : 6,
    '1L' : 7,
    '2L' : 8,
    '3L' : 9,
    '4L' : 10,
    '5L' : 11
}


# In[ ]:


temp = []
for label in y_train:
    temp.append(label_to_int[label])
y_train = temp.copy()

temp = []
for label in y_test:
    temp.append(label_to_int[label])
y_test = temp.copy()


# In[ ]:


y_train = keras.utils.to_categorical(y_train, num_classes = 12)
y_test = keras.utils.to_categorical(y_test, num_classes = 12)


# ## Creating the network

# In[ ]:


weight_decay = 1e-4

num_classes = 12

model = Sequential()

model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(128,128,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
 
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (4,4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(128, activation="linear"))
model.add(Activation('elu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0003), metrics=['accuracy'])
 
model.summary()


# In[ ]:


model.fit(x = X_train,y = y_train, batch_size=64, validation_data = (X_test,y_test), epochs = 5)

