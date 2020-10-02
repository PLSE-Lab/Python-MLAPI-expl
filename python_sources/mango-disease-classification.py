#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# In[ ]:


import pathlib

dir1 = '../input/Mango'
PATH = os.path.dirname(dir1)
print(PATH)


# In[ ]:


#disease = 'plant_disease/Mango_(P0)/diseased'#os.path.join(PATH, "diseased")
#health = 'plant_disease/Mango_(P0)/healthy' #os.path.join(PATH, 'healthy')
train_dir = (dir1 +'/train')
val_dir = (dir1 +'/val')
test_dir = (dir1 +'/test')


# In[ ]:



train_healthy = (train_dir + '/healthy')
train_diseased = (train_dir + '/diseased')
val_healthy = (val_dir + '/healthy')
val_diseased = (val_dir + '/diseased')

test_healthy = (test_dir + '/healthy')
test_diseased = (test_dir + '/diseased')


# In[ ]:


num_diseased_tr = len(os.listdir(train_diseased))
num_healthy_tr = len(os.listdir(train_healthy))
num_diseased_val = len(os.listdir(val_diseased))
num_healthy_val = len(os.listdir(val_healthy))
print(num_healthy_tr)
print(num_diseased_tr)

total_train = num_diseased_tr + num_healthy_tr
total_validation = num_diseased_val + num_healthy_val


# ### Image Generator

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_size = 150

image_gen_train = ImageDataGenerator(rescale = 1./255,
                                     rotation_range = 45,
                                     width_shift_range=.15,
                                     height_shift_range =.15,
                                     horizontal_flip=True,
                                     zoom_range=0.5)


# In[ ]:


# x_train, x_test, y_train, y_test = train_test_split(disease, health, test_size=0.2, random_state=289)
train_data_gen = image_gen_train.flow_from_directory(batch_size=32,
                                              directory= train_dir,
                                              shuffle=True,
                                              target_size=(image_size, image_size),
                                              class_mode='binary')  


# In[ ]:


image_gen_val = ImageDataGenerator(rescale = 1./255)


# In[ ]:


val_data_gen = image_gen_val.flow_from_directory(batch_size=32,
                                              directory= val_dir,
                                              shuffle=True,
                                              target_size=(image_size, image_size),
                                              class_mode='binary')  


# ### Convolution network

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt

model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(image_size, image_size ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# In[ ]:


model_new.compile(optimizer='adam',
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=['accuracy'])

model_new.summary()


# ### Fit the created model

# In[ ]:


epochs = 15
history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // 32,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_validation // 32
)


# ### Plot loss and accuracy

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:




