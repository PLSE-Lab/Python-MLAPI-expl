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

# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir('../input/train_images')


# In[ ]:


train=pd.read_csv('../input/train.csv')


# In[ ]:


train.diagnosis=train.diagnosis.astype('str')
train.id_code=train.id_code+'.png'
train.head()


# In[ ]:


train.diagnosis.value_counts()


# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=0.2)


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('../input/train_images/0125fbd2e791.png')
imgplot = plt.imshow(img)
plt.show()
print(img.shape)


# In[ ]:


train_generator=train_datagen.flow_from_dataframe(
        dataframe=train,
        directory='../input/train_images/',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(256,256),
        class_mode='categorical',
        subset='training'
        )

valid_generator=train_datagen.flow_from_dataframe(
        dataframe=train,
        directory='../input/train_images/',
        x_col='id_code',
        y_col='diagnosis',
        target_size=(256,256),
        class_mode='categorical',
        subset='validation'
        )


# Training the Model

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten


# In[ ]:


model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


# In[ ]:


history=model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=valid_generator,
        steps_per_epoch=train_generator.n,
        validation_steps=valid_generator.n
)


# In[ ]:




