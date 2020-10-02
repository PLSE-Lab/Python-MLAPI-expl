#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
       os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Chest Xray Classification - NORMAL and PNEUMONIA Cases
# __This jupiter book uses the Convolutional Neural network and is trained with Normal and Pneumonia xray iamges. The built model should successfully classify normal xray image and pneumonia ones__

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2


# In[ ]:


#let us plot some images from train path - normal
train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/'
images = os.listdir(train_path)
fig = plt.figure(figsize=(10, 10))
for i in range(0,9):
    plt.subplot(3,3,i+1)
    plt.imshow(plt.imread(train_path+images[i]), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(images[i])
    


# In[ ]:


#let us plot some images from train path - pneumonia
train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/'
images = os.listdir(train_path)
fig = plt.figure(figsize=(10, 10))
for i in range(0,9):
    plt.subplot(3,3,i+1)
    plt.imshow(plt.imread(train_path+images[i]), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(images[i])


# In[ ]:


#check the shape of the images
X_train=[]

y_train=[]


# In[ ]:


X_test=[]
y_test=[]


# In[ ]:


train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/'
images = os.listdir(train_path)
for image in images:
    if "jpeg" in image:
        img = cv2.imread(train_path+image)
        image = cv2.resize(img,(64,64))
        X_train.append(image)
        y_train.append(0)
    


# In[ ]:


train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/'
images = os.listdir(train_path)
for image in images:
    if "jpeg" in image:
        img = cv2.imread(train_path+image)
        image = cv2.resize(img,(64,64))
        X_train.append(image)
        y_train.append(1)
    


# In[ ]:


test_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/'
images = os.listdir(test_path)
for image in images:
    if "jpeg" in image:
        img = cv2.imread(test_path+image)
        image = cv2.resize(img,(64,64))
        X_test.append(image)
        y_test.append(0)
    


# In[ ]:


test_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/'
images = os.listdir(test_path)
for image in images:
    if "jpeg" in image:
        img = cv2.imread(test_path+image)
        image = cv2.resize(img,(64,64))
        X_test.append(image)
        y_test.append(1)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, AveragePooling2D, Dense,BatchNormalization,MaxPooling2D,MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# In[ ]:


X_train[0].shape


# In[ ]:


import numpy as np
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


# In[ ]:


#y_train = to_categorical(y_train,num_classes=2)
#y_test = to_categorical(y_test,num_classes=2)


# In[ ]:


model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(64,64,3)))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(filters=3, kernel_size=(14,14), strides=(1,1), activation='relu'))
model.add(AveragePooling2D(pool_size=(5,5), strides=(1,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=4, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(820, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(54, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(27, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train,y_train, epochs=50, validation_data=(X_test,y_test))


# In[ ]:


import pandas as pd
history_df = pd.DataFrame(history.history)


# In[ ]:


history_df.boxplot()


# In[ ]:


y_predictions = model.predict(X_test).argmax(1)


# In[ ]:


index = int(input('Enter a index to test the prediction  '))
plt.imshow(X_test[index]);
plt.tight_layout()
actual=''
predicted=''
if(y_predictions[index]==0):
  predicted='NORMAL'
else: 
  predicted = 'PNEUMONIA'
  
if(y_test[index]==0):
  actual='NORMAL'
else: 
  actual = 'PNEUMONIA'
   
  
plt.title("Predicted  ==> "+predicted +" \nActual ==>"+ actual )
plt.xticks([])
plt.yticks([])


# In[ ]:




