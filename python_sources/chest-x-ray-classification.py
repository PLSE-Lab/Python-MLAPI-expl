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


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Activation, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', input_shape=(64,64,1), activation='relu'))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.40))
model.add(Dense(2, activation='softmax'))


# In[ ]:


gen = ImageDataGenerator()
train_batches = gen.flow_from_directory("../input/chest_xray/chest_xray/train",model.input_shape[1:3],color_mode="grayscale",shuffle=True,seed=1,
                                        batch_size=16)
valid_batches = gen.flow_from_directory("../input/chest_xray/chest_xray/val", model.input_shape[1:3],color_mode="grayscale", shuffle=True,seed=1,
                                        batch_size=16)
test_batches = gen.flow_from_directory("../input/chest_xray/chest_xray/test", model.input_shape[1:3], shuffle=False,
                                       color_mode="grayscale", batch_size=8)


# In[ ]:


model.compile(Adam(lr=0.001),loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


model.fit_generator(train_batches,validation_data=valid_batches,epochs=30, steps_per_epoch=16, validation_steps=16)


# In[ ]:


no_steps = len(test_batches)
p = model.predict_generator(test_batches, steps=no_steps, verbose=True)
pre = pd.DataFrame(p)


# In[ ]:


pre["filename"] = test_batches.filenames
pre["label"] = (pre["filename"].str.contains("PNEUMONIA")).apply(int)
pre['pre'] = (pre[1]>0.5).apply(int)


# In[ ]:


accuracy_score(pre["label"], pre["pre"])


# In[ ]:


# We get an accuracy score of 78.5% !!!


# In[ ]:





# In[ ]:




