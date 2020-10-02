#!/usr/bin/env python
# coding: utf-8

# # CNN for classifying letters 
# By: Hesham Asem
# 
# ______
# 
# here we'll build a Conv2d to be used in reading & classifying about half million pictures of  first 10 alphabetic letters . . 
# 
# you can find data file  here :  https://www.kaggle.com/jwjohnson314/notmnist
# 
# let;s first import libraries 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import keras
import os
import glob as gb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten ,Conv2D, MaxPooling2D


# then we need to check the folders to know what letters available . 

# In[ ]:


all_letters = os.listdir('../input/notmnist/notMNIST_large/notMNIST_large')

print(f'We have {len(all_letters)} letters , which are : {all_letters}')


# ____
# 
# 
# # Read Data . 
# 
# we'll use glob library to collect all png pictures to know how many pictures we have for each letter . 

# In[ ]:


total_images = 0
for letter in all_letters : 
    available_images = gb.glob(pathname= f'../input/notmnist/notMNIST_large/notMNIST_large/{letter}/*.png')
    total_images+=len(available_images)
    print(f'for letter {letter} we have  {len(available_images)} available images')
print('-----------------------')    
print(f'Total Images are {total_images} images')


# total 529 thousand images for all 10 letters , now let's create X & y variables , so we can fill them with read data
# 

# In[ ]:


X = list(np.zeros(shape=(total_images , 28,28)))
y = list(np.zeros(shape=(total_images)))


# now to open each file & read it using plt.imread , then fill it in its place in X & y data

# In[ ]:


i=0
y_value = 0
for letter in all_letters : 
    available_images = gb.glob(pathname= f'../input/notmnist/notMNIST_large/notMNIST_large/{letter}/*.png')
    for image in available_images : 
        try : 
            x = plt.imread(image)
            X[i] = x
            y[i] = y_value
            i+=1
        except : 
            pass
    y_value+=1


# ____
# 
# # Forming Dimensions
# 
# since (y) data now is a single number vary from 0 to 9 , we'll need to categorize it using OneHotEncoder from sklearn , so it be ready for the softmax activation functing in CNN

# In[ ]:


ohe  = OneHotEncoder()
y = np.array(y)
y = y.reshape(len(y), 1)
ohe.fit(y)
y = ohe.transform(y).toarray()


# now we can check a random y value

# In[ ]:


y[10000]


# then we have to expand X dimension to be suitable with the CNN dimensions

# In[ ]:


X = np.expand_dims(X, -1).astype('float32')/255.0


# now X shape should be : sample size * 28 * 28 * 1

# In[ ]:


X.shape


# # Splitting Data
# 
# now lets split our dat to Train , Cross-Validation & Test sets . . 
# 
# first to create X_part & y_part which is 85% of data , also X_test & y_test which is 15%
# 

# In[ ]:


X_part, X_cv, y_part, y_cv = train_test_split(X, y, test_size=0.15, random_state=44, shuffle =True)

print('X_train shape is ' , X_part.shape)
print('X_test shape is ' , X_cv.shape)
print('y_train shape is ' , y_part.shape)
print('y_test shape is ' , y_cv.shape)


# then to split X_part & y_part into train & test

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.25, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# # Build the Model
# 
# now let's build the model with Keras , using Conv2d & Maxpooling tools , & not to forget to dropout some cells to avoid OF

# In[ ]:


KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(filters = 32, kernel_size = 4,  activation = tf.nn.relu , padding = 'same'),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=None, padding='valid'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=32, kernel_size=4,activation = tf.nn.relu , padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=64, kernel_size=5,activation = tf.nn.relu , padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),    
        keras.layers.Dropout(0.5),        
        keras.layers.Dense(64),    
        keras.layers.Dropout(0.3),            
        keras.layers.Dense(units= 10,activation = tf.nn.softmax ),                

    ])
    

KerasModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# then to train it , using few number of epochs to avoid OF

# In[ ]:


#Train
KerasModel.fit(X_train,y_train,validation_data=(X_cv, y_cv),epochs=3,batch_size=64,verbose=1)


# now how the model looks like ? 

# In[ ]:


KerasModel.summary()


# # Predicting
# 
# then we'll predict X_test

# In[ ]:


y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))


# and we can check random samples from X_test

# In[ ]:


Letters ={0:'A', 1:'B' , 2:'C' ,3:'D' ,4:'E' ,5:'F' ,6:'G' ,7:'H' ,8:'I' ,9:'J' }

for i in list(np.random.randint(0,len(X_test) ,size= 10)) : 
    print(f'for sample  {i}  the predicted value is   {Letters[np.argmax(y_pred[i])]}   , while the actual letter is {Letters[np.argmax(y_test[i])]}')


# & to measure the loss & accuracy

# In[ ]:


ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))


# great , with 91% accuracy we achieved good result without moving to OF

# In[ ]:




