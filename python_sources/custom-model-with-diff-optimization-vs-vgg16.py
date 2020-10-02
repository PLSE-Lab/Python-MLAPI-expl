#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd ..


# In[ ]:


cd input/chest-xray-pneumonia/chest_xray/


# In[ ]:


train_dir = "train/"
val_dir = "test/"
test_dir = "val/"


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1.0/255.)
val_datagen = ImageDataGenerator(rescale = 1./255.)


# In[ ]:


train_generator = train_datagen.flow_from_directory(train_dir, target_size = (150,150), class_mode = "binary", batch_size=32)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(150,150), class_mode = "binary", batch_size = 16)
test_generator = val_datagen.flow_from_directory(test_dir, target_size=(150,150))


# In[ ]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(256, (3,3),input_shape = (150,150,3), activation='relu', kernel_regularizer = tf.keras.regularizers.l2()),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(128, (3,3),activation = "relu", kernel_regularizer = tf.keras.regularizers.l2()),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    
    tf.keras.layers.Conv2D(64,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    
    tf.keras.layers.Conv2D(32,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, "sigmoid")
])


# In[ ]:


optimizer = tf.keras.optimizers.SGD(lr = 1e-4, momentum=0.99)


# In[ ]:


model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ['acc'])


# In[ ]:


history = model.fit(train_generator, validation_data=val_generator, epochs = 10, verbose = 1)


# In[ ]:


history = model.fit(train_generator, validation_data=val_generator, epochs = 5, verbose = 1)


# In[ ]:


plt.plot(history.history['acc'], label = "training Accuracy")
plt.plot(history.history['val_acc'], label = "Validation Accuracy")
plt.legend()


# In[ ]:


plt.plot(history.history['loss'], label = "Train loss")
plt.plot(history.history['val_loss'], label = "Val loss")
plt.legend()


# ## Tuning optimizer 

# In[ ]:


optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.99)


# In[ ]:


model.compile(optimizer = optimizer1, loss = "binary_crossentropy", metrics = ['acc'])


# In[ ]:


lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-8 * 10**(epochs*0.6))


# In[ ]:


history = model.fit(train_generator, validation_data=val_generator, epochs = 10, callbacks=[lr_schedule], verbose = 1, steps_per_epoch=75)


# In[ ]:


x = 1e-8 * 10**(np.arange(10) * 0.6)  
y = history.history['loss']
plt.semilogx(x,y)


# ## RMSprop 

# In[ ]:


optimizer2 = tf.keras.optimizers.RMSprop(learning_rate=1e-8, momentum=0.99)


# In[ ]:


model.compile(optimizer = optimizer1, loss = "binary_crossentropy", metrics = ['acc'])


# In[ ]:


history = model.fit(train_generator, validation_data=val_generator, epochs = 10, callbacks=[lr_schedule], verbose = 1, steps_per_epoch=50)


# In[ ]:


x = 1e-8 * 10**(np.arange(10) * 0.6)  
y = history.history['loss']
plt.semilogx(x,y)


# ## Adam 

# In[ ]:


optimizer3 = tf.keras.optimizers.Adam(lr = 0.001)


# In[ ]:


model.compile(optimizer = optimizer3, loss = "binary_crossentropy", metrics = ['acc'])


# In[ ]:


history = model.fit(train_generator, validation_data=val_generator, epochs = 10, callbacks=[lr_schedule], verbose = 1, steps_per_epoch=50)


# In[ ]:


x = 1e-8 * 10**(np.arange(10) * 0.6)  
y = history.history['loss']
plt.semilogx(x,y)


# ## VGG16 

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout


# In[ ]:


base_model = tf.keras.applications.VGG16(include_top = False, weights = 'imagenet', input_shape = (240,240,3))


# In[ ]:


base_model.trainable = False


# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# In[ ]:


pred_layer = tf.keras.layers.Dense(1, "sigmoid")


# In[ ]:


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  pred_layer
])


# In[ ]:


model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ['acc'])


# In[ ]:


history2 = model.fit(train_generator, validation_data=val_generator, epochs = 10)


# In[ ]:


plt.plot(history2.history['acc'], label = "training Accuracy")
plt.plot(history2.history['val_acc'], label = "Validation Accuracy")
plt.legend()


# In[ ]:


plt.plot(history.history['loss'], label = "Train loss")
plt.plot(history.history['val_loss'], label = "Val loss")
plt.legend()


# In[ ]:




