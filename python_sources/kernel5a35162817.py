#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras


# In[ ]:


train_data=sio.loadmat('../input/svhndataset/extra_32x32.mat')
test_data=sio.loadmat('../input/svhndataset/test_32x32.mat')


# In[ ]:


X_train, y_train=train_data['X'],train_data['y']
X_test, y_test=test_data['X'],test_data['y']


# In[ ]:


# 10 class uisng for number '0'
np.unique(y_train)


# In[ ]:


X_train.shape


# In[ ]:


# for calling one image need traspose data
X_train=X_train.transpose((3,0,1,2))
X_test=X_test.transpose((3,0,1,2))


# In[ ]:


for i in range(2):
    plt.subplots()
    plt.imshow(X_train[i])
    plt.title(y_train[i])


# In[ ]:


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters = 3,
                        kernel_size=(3,3),
                        input_shape = (X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2]),
                        activation='relu',
                        padding='same'))

model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256,activation='relu'))


model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(256,activation='relu'))

model.add(keras.layers.Dense(11, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


y_train.shape


# In[ ]:


#One hot encoder
from keras.utils import np_utils
y_train_labels = np_utils.to_categorical(y_train)


# In[ ]:


#model.fit(X_train, y_train_labels, batch_size=128, epochs=5, validation_split=0.2)


# In[ ]:


#y_pred=model.predict_classes(X_test)


# In[ ]:


#Accuracy for CNN model - first variant
#accuracy_score(y_test,y_pred)


# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model


# In[ ]:


# Use architect from ResNet50
base_model = ResNet50(include_top = False,
                   weights = 'imagenet',
                   input_shape = (32, 32, 3))


# In[ ]:


# Dont fix layers for training
for layer in base_model.layers:
    layer.trainable = True 


# In[ ]:


x = base_model.layers[-2].output

x = keras.layers.Flatten()(x)

x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(512, activation='relu')(x)

x = keras.layers.Dense(11, activation='softmax')(x)

model2 = Model(inputs=base_model.input, outputs=x)

model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model2.summary()


# In[ ]:


model2.fit(X_train, y_train_labels, batch_size=300, epochs=10, validation_split=0.2)


# In[ ]:


#Accuracy for CNN model - second variant. With structure ResNet50
y_pred2=model2.predict(X_test)


# In[ ]:


#Convert to classes
y_pred2_cls=[]
for i in range(y_pred2.shape[0]):
    y_pred2_cls.append(np.argmax(y_pred2[i]))


# In[ ]:


accuracy_score(y_test,y_pred2_cls)


# In[ ]:


for i in range(3):
    print('Predict value = ',y_pred2_cls[i],"    True value =",y_test[i] )
    plt.subplots()
    plt.imshow(X_test[i])
    plt.title(y_test[i])    


# In[ ]:




