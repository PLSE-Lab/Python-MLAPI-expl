#!/usr/bin/env python
# coding: utf-8

# # Malaria Cell Image Classification using CNN
# 
# This kernel is a demonstration of using CNN with keras on tensorflow to classify if an image of a cell has Malaria or not.

# In[ ]:


import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np


# ### Data Augmentation
# Here, I use `ImageDataGenerator` from keras.

# In[ ]:


datagen = ImageDataGenerator(
    rescale = 1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip = True,
    validation_split=0.1
)

batch_size = 48
num_classes = 2
image_size = 64


# ### Input Data
# Using `flow_from_directory` from keras lets us easily import the images into our script. It automatically lables the images according to the folder names.

# In[ ]:


train_generator = datagen.flow_from_directory(
    '../input/cell_images/cell_images',
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'binary',
    subset='training'
)

dev_generator = datagen.flow_from_directory(
    '../input/cell_images/cell_images',
    target_size = (image_size, image_size),
    batch_size = batch_size,
    class_mode = 'binary',
    subset='validation'
)


# ### Visualize
# Let us inspect an image from the dataset.

# In[ ]:


sample = train_generator.next();
plt.imshow(sample[0][0])
train_generator.reset()


# ### Define the Model
# The model is as follows.

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


# ### Time to Train!

# In[ ]:


history=model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=60,
    validation_data=dev_generator,
    validation_steps=800 // batch_size)


# ### Visualize the results

# In[ ]:


metrics = history.history

plt.subplot(211)

plt.plot(metrics['acc'],color='blue')
plt.plot(metrics['val_acc'],color='green')

plt.subplot(212)

plt.plot(metrics['loss'],color='yellow')
plt.plot(metrics['val_loss'],color='red')


# In[ ]:




