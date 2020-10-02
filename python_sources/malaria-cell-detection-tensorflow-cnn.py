#!/usr/bin/env python
# coding: utf-8

# ### Author : Sanjoy Biswas
# ### Topic : Malaria Cell Detection Using CNN
# ### Email : sanjoy.eee32@gmail.com

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import Libraries

# In[ ]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History


# # Data Augmentation

# In[ ]:


datagen = ImageDataGenerator(rescale=1.0/255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, validation_split = 0.1)

batch_size = 48
num_classes = 2
image_size = 64


# In[ ]:





# # Input Data

# In[ ]:


train_generator = datagen.flow_from_directory(
    '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images',
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'binary',
    subset='training'
)

dev_generator = datagen.flow_from_directory(
    '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images',
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'binary',
    subset='validation'
)


# In[ ]:


sample = train_generator.next();
plt.imshow(sample[0][0])
train_generator.reset()


# # Define Model

# In[ ]:


model = Sequential()
model.add(Conv2D(64,(3,3)
        ,input_shape=(image_size,image_size,3)
        ,activation='relu'))

model.add(Conv2D(64,(3,3)
        ,input_shape=(image_size,image_size,3)
        ,activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3)
        ,input_shape=(image_size,image_size,3)
        ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3)
        ,input_shape=(image_size,image_size,3)
        ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


# # Train Model

# In[ ]:


history=model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=60,
    validation_data=dev_generator,
    validation_steps=800 // batch_size)


# # Visualize the results

# In[ ]:


metrics = history.history



plt.subplot(212)

plt.plot(metrics['loss'],color='yellow')
plt.plot(metrics['val_loss'],color='red')


# In[ ]:




