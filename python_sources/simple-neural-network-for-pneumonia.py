#!/usr/bin/env python
# coding: utf-8

# Specifiy train, test, val directory path

# In[ ]:


train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/' 
test_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'
val_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'


# In[ ]:


import tensorflow as tf 
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMPSprop


# Using features of ImageDatagenerator that can automatically load data and determine the class based on folder name. 

# In[ ]:


IMG_SIZE = (128, 128)
BATCH_SIZE = 128

test_image_generator = image.ImageDataGenerator(rescale=1./255)
train_image_generator = image.ImageDataGenerator(rescale=1./255)
val_image_generator = image.ImageDataGenerator(rescale=1./255)

train_data_generator = train_image_generator.flow_from_directory(
    directory=train_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_data_generator = test_image_generator.flow_from_directory(
    directory=test_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

val_data_generator = val_image_generator.flow_from_directory(
    directory=val_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

Make sure labels for train, test, val correctly 
# In[ ]:


print(train_data_generator.class_indices,
test_data_generator.class_indices,
val_data_generator.class_indices)


# Build simple CNN model (note: it's really simple, maybe not really effective and kinda overfitt) 

# In[ ]:


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  padding='same',
                  input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='softmax')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc']
)

model.summary()


# In[ ]:


history = model.fit(
    train_data_generator,
    epochs=15,
    validation_data=val_data_generator,
    verbose=1)


# In[ ]:


prediction = model.evaluate(test_data_generator)


# In[ ]:





# In[ ]:




