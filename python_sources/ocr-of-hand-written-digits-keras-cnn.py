#!/usr/bin/env python
# coding: utf-8

# Developing a CNN model with already available dataset in Keras DL library.

# In[ ]:


#data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical


# In[ ]:


# Details of dataset
# http://yann.lecun.com/exdb/mnist/
# from keras.datasets import mnist
# (X_train,y_train) , (X_test,y_test) = mnist.load_data()


# Unable to use the dataset from Keras library - Got to know that kaggle kernels do not have network access https://www.kaggle.com/data/50566
# 
# Loading data from the Digit Recognizer competition dataset
# https://www.kaggle.com/c/digit-recognizer/kernels.

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print('train_set shape : {}, test_set shape : {}'.format(train.shape,test.shape))


# In[ ]:


print(train.head(3))
print("--------------------------------------")
print(test.head(3))


# We have 42k labelled images for training and validation, 28k images as unlabelled test set. 
# 
# I guess the test set is not labelled as the dataset is part of competition. Will have to split the given train set to test set too, or may be will add this note to the competition.! As of now will proceed considering the whole train data for** both training and testing**

# In[ ]:


X_train = train.drop("label",axis=1)
Y_train = to_categorical(train.label)


# In[ ]:


#Normalization
X_train = X_train / 255.0
Y_train = Y_train / 255.0


# **Baseline model with Multi-layer Perceptrons **

# In[ ]:


blx_tr, blx_val, bly_tr, bly_val = train_test_split(X_train, Y_train, test_size=0.17, random_state=346)


# In[ ]:


baseline = Sequential()
baseline.add(Dense(X_train.values.shape[1] , input_dim=X_train.values.shape[1], activation = 'relu'))
baseline.add(Dense(256, activation = 'relu'))
baseline.add(Dropout(0.2))
baseline.add(Dense(64, activation = 'relu'))
baseline.add(Dense(10, activation = 'softmax'))
baseline.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


base_line = baseline.fit(blx_tr,bly_tr, validation_data=(blx_val,bly_val), batch_size=100,epochs=10)


# Validation accuracy is good at 97.56%
# 
# The validation accuracy is greater than the training accuracy almost every time during the training. That means that **our model dosen't not overfit the training set.**

# ** Model using CNN architectrue **

# In[ ]:


#CNN
X_tr = X_train.values.reshape(-1,28,28,1)
Y_tr = Y_train


# In[ ]:


x_tr,x_val,y_tr,y_val = train_test_split(X_tr,Y_tr,test_size=0.1,random_state=346)


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape=(28,28,1), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(56, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model = model.fit(x_tr,y_tr,validation_data=(x_val,y_val),batch_size=100,epochs=10)


# Validation accuracy is greater than training accuracy, hence porceeding with this model, this model 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




