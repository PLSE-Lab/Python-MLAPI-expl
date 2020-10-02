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
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


img_width,img_height = 224,224
train_data_dir='../input/car-classificationproject-vision/Train/Train'
validation_data_dir='../input/car-classificationproject-vision/Test/Test'
batch_size =15

datagen = ImageDataGenerator(rescale = 1./255)


train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True)  

test_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width,img_height), # 120, 265 original
    batch_size = batch_size,
    class_mode = 'categorical')


# In[ ]:



model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.3))



model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(45, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


from keras.utils import plot_model
plot_model(model,to_file='model.png')


# In[ ]:




model.fit_generator(
    train_generator,
    steps_per_epoch = 270,
    epochs = 30,
    validation_data = test_generator,
)


# In[ ]:





# In[ ]:




