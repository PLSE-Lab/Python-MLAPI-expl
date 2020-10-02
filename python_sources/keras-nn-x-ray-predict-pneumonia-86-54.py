#!/usr/bin/env python
# coding: utf-8

# # Binary classification with Keras neural network

# English is not my native language, so sorry for any mistake.
# 
# If you like my Kernel, give me some feedback and also votes up my kernel.

# ### Import

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import os
import numpy as np
import pandas as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Img

# In[ ]:


print(os.listdir("../input/chest_xray/chest_xray"))

print(os.listdir("../input/chest_xray/chest_xray/train"))

print(os.listdir("../input/chest_xray/chest_xray/train/"))


# In[ ]:


img_name = 'NORMAL2-IM-0588-0001.jpeg'
img_normal = load_img('../input/chest_xray/chest_xray/train/NORMAL/' + img_name)

print('NORMAL')
plt.imshow(img_normal)
plt.show()


# In[ ]:


img_name = 'person63_bacteria_306.jpeg'
img_pneumonia = load_img('../input/chest_xray/chest_xray/train/PNEUMONIA/' + img_name)

print('PNEUMONIA')
plt.imshow(img_pneumonia)
plt.show()


# ### Create variable

# In[ ]:


# dimensions of our images.
img_width, img_height = 150, 150


# In[ ]:


train_data_dir = '../input/chest_xray/chest_xray/train'
validation_data_dir = '../input/chest_xray/chest_xray/val'
test_data_dir = '../input/chest_xray/chest_xray/test'

nb_train_samples = 5217
nb_validation_samples = 17
epochs = 20
batch_size = 16


# In[ ]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# ### Create Sequential model

# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# ### Check information about model

# In[ ]:


model.layers


# In[ ]:


model.input


# In[ ]:


model.output


# ### Compile

# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# ### Upload img

# In[ ]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[ ]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[ ]:


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# In[ ]:


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# ### Fit model

# In[ ]:


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# ### Save model`s weights

# In[ ]:


model.save_weights('first_try.h5')


# In[ ]:


# evaluate the model
scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

