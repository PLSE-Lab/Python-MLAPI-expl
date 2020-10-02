#!/usr/bin/env python
# coding: utf-8

# # CNN with OpenCV
# **In this kernel I am going to classify x-ray images in order to understand whether the person has Pneumonia or not.Firstly , i will going to read my images with openCV.Then I will preprocesses my data, create my model.At last, I will feed my images to my Convolutional Neural Network, and calculate my accuracy.*
# 
# Let's start with importing our libraries.
# 
# **Content:**
# 1. [Data Loading with OpenCV](#1)
# 1. [Data Preprocessing](#2)
# 1. [Creating Model with Keras](#3)
# 1. [Data Augmentation](#4)
# 1. [Training](#5)
# 1. [Prediction and Accuracy](#6)
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # reading data
import cv2 # reading images
import matplotlib.pyplot as plt
# /kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/


# <a id="1"></a>
# # 1.Data Loading with OpenCV

# First of all i need to get my jpeg files to arrays but i have 3875 pneumonia and 1341 normal data.I need to delete some of pneumonia data.

# In[ ]:


datadir="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/"
categories = ["NORMAL","PNEUMONIA"]
training_data =[]
num=0
for category in categories:
    path = os.path.join(datadir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(200,200))
            if(num<=2642): # i have more than i need pneumonia data so i added an if condition
                training_data.append([new_array,class_num])
                num+=1
            else:
                break
        except Exception:
            pass


# I added all my jpeg files into training_data above as you can see.But i still got my last image in new_array variable.Let's see what is it look like.

# In[ ]:


plt.imshow(new_array,cmap="gray")
plt.show()


# Now i need to import my validation data too.

# In[ ]:


datadir="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/"
categories = ["NORMAL","PNEUMONIA"]
test_data =[]
num=0
for category in categories:
    path = os.path.join(datadir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(200,200))
            if(num<=2642): # i have more than i need pneumonia data so i added an if condition
                test_data.append([new_array,class_num])
                num+=1
            else:
                break
        except Exception:
            pass


# In[ ]:


plt.imshow(new_array,cmap="gray")
plt.show()


# <a id="2"></a>
# # 2.Data Preprocessing

# Now i need to shuffle my data to learn better.

# In[ ]:


len(training_data)
import random
random.shuffle(training_data)
x_train=[]
y_train=[]
x_test=[]
y_test=[]


# I need to save my labels and features in diferrent arrays and then reshape my train and validation datas.

# In[ ]:


for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train = np.array(x_train).reshape(-1,200,200,1)
#I convert numpy and then i added 1 at the end because keras need 3 
x_train.shape
x_train = x_train/255.0 # normalization


# In[ ]:


for features, label in test_data:
    x_test.append(features)
    y_test.append(label)
x_test = np.array(x_test).reshape(-1,200,200,1)
#I convert numpy and then i added 1 at the end because keras need 3 
x_test.shape
x_test = x_test/255.0 # normalization


# My datas are almost ready.Now i need to convert my y_train and y_test to one hot vector.

# In[ ]:


from keras.utils.np_utils import to_categorical 
y_train = to_categorical(y_train, num_classes = 2)
y_test = to_categorical(y_test, num_classes = 2)


# As you can see my data is shuffled.

# In[ ]:


for a in y_train[10:20]:
    print(a)


# <a id="3"></a>
# # 3.Creating Model with Keras

# Now I am going to create my model.For this I will follow this structure :
# 
# Conv -> MaxPool -> Dropout -> Conv -> MaxPool -> Dropout -> Conv -> MaxPool -> Dropout -> Fully Connected Layer

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(20,20), padding='Same', activation='relu', input_shape=(200,200,1)))
model.add(MaxPool2D((10,10),strides=(1,1)))
model.add(Conv2D(filters=6, kernel_size=(7,7), padding='Same', activation='relu'))
model.add(MaxPool2D((10,10),strides=(1,1)))
#model.add(MaxPool2D((2,2),strides=(1,1)))
#model.add(Conv2D(filters=8, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='sigmoid'))


# 

# <a id="5"></a>
# # 5.Training

# I am going to use Adam optimizer.

# In[ ]:


optimizer = Adam(lr=0.0000008)
model.compile(optimizer = optimizer , loss ="binary_crossentropy", metrics=["accuracy"])

epochs = 20
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=5,batch_size=16)


# <a id="6"></a>
# # 6.Accuracy

# Time for plot our test loss and confusion matrix.

# In[ ]:


plt.plot(history.history["val_loss"], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


import seaborn as sns
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1) 
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# **As you can see my model is not so good but you can work on hyperparameters and see how it works !**

# In[ ]:


predictions = model.predict(x_test)
score = model.evaluate(x_test,y_test,verbose=0)
print("Test loss :",score[0])
print("Test Accuracy : ",score[1])


# In[ ]:




