#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[ ]:


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D


# # Create Data Directory Variables And List of Labels

# In[ ]:


DATADIR_Train = '../input/fingers/train'
DATADIR_Test = '../input/fingers/test'
IMG_SIZE = 128
label_list = ['0L', '1L', '2L', '3L', '4L', '5L', '0R', '1R', '2R', '3R', '4R', '5R']


# # Fetch Training and Testing Datasets

# In[ ]:


training_data = []
test_data = []


for train_img in os.listdir(DATADIR_Train):
    label_str = train_img[-5: -7: -1][::-1]
    label = label_list.index(label_str)
    img = cv2.imread(os.path.join(DATADIR_Train, train_img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    training_data.append([img, label])
    
for test_img in os.listdir(DATADIR_Test):
    label_str = test_img[-5: -7: -1][::-1]
    label = label_list.index(label_str)
    img = cv2.imread(os.path.join(DATADIR_Test, test_img), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_data.append([img, label])


# # Print Length of Datasets

# In[ ]:


print("Training Data :",len(training_data))
print("Test Data :",len(test_data))


# # Check Training Data

# In[ ]:


plt.imshow(training_data[0][0], cmap = 'gray')
print(label_list[training_data[0][1]])


# # Check Test Data

# In[ ]:


plt.imshow(test_data[0][0], cmap = 'gray')
print(label_list[test_data[0][1]])


# # Check Shape of Training Data

# In[ ]:


training_data[0][0].shape


# # Shuffle Training and Testing Data

# In[ ]:


random.shuffle(training_data)
random.shuffle(test_data)


# # Separate Feature and Label from Training and Testing Data

# In[ ]:


x_train = []
y_train = []
x_test = []
y_test = []

for feature, label in training_data:
    x_train.append(feature)
    y_train.append(label)
    
for feature, label in test_data:
    x_test.append(feature)
    y_test.append(label)


# # Check Length of Features and Labels

# In[ ]:


print("x_train : ", len(x_train))
print("y_train : ", len(y_train))
print("x_test : ", len(x_test))
print("y_test : ", len(y_test))


# # Reshape Image

# In[ ]:


x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


y_train = np.array(y_train)
y_test = np.array(y_test)


# # Scaling Image

# In[ ]:


x_train = x_train/255.0
x_test = x_test/255.0


# # Define Model using tensorflow keras

# In[ ]:


model = Sequential()

# First Layer
model.add(   Conv2D(64,  (3, 3), input_shape = x_train.shape[1:])   )
model.add( Activation('relu') )
model.add( MaxPool2D(pool_size = (2,2)) )

# Second Layer
model.add(   Conv2D(64,  (3, 3))   )
model.add( Activation('relu') )
model.add( MaxPool2D(pool_size = (2,2)) )

# Third Layer
model.add(Flatten())
model.add(Dense(64))
model.add( Activation('relu') )

# Output Layer
model.add(Dense(12))
model.add(Activation('sigmoid'))


# # Compile Model

# In[ ]:


model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = 'RMSprop',
             metrics = ['accuracy'])


# # Train Model

# In[ ]:


model.fit(x_train, y_train, batch_size = 32, epochs=3, validation_data = (x_test, y_test))


# # Save Model

# In[ ]:


model.save('Fingers_Detection_CNN_Tensorflow_Keras.model')

