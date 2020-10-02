#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
print('Tensorflow verions:', tf.__version__)


# In[ ]:


base_dir = '/kaggle/input/butterfly-images/butterflies'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')

train_mj_dir = os.path.join(train_dir, 'maniola_jurtina')  
train_pt_dir = os.path.join(train_dir, 'pyronia_tithonus') 
validation_mj_dir = os.path.join(validation_dir, 'maniola_jurtina')
validation_pt_dir = os.path.join(validation_dir, 'pyronia_tithonus')


# In[ ]:


num_mj_tr = len(os.listdir(train_mj_dir))
num_pt_tr = len(os.listdir(train_pt_dir))

num_mj_val = len(os.listdir(validation_mj_dir))
num_pt_val = len(os.listdir(validation_pt_dir))

total_train = num_mj_tr + num_pt_tr
total_val = num_mj_val + num_pt_val


# In[ ]:


train_image_generator = ImageDataGenerator(rescale=1./255.0,
                                           horizontal_flip=True,
                                           zoom_range=0.5,
                                           rotation_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255.0)


# In[ ]:


IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10


# In[ ]:


train_batches = train_image_generator.flow_from_directory(train_dir,
                                                           target_size=(IMG_SIZE,IMG_SIZE),
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True)

valid_batches = validation_image_generator.flow_from_directory(validation_dir,
                                                           target_size=(IMG_SIZE,IMG_SIZE),
                                                           batch_size=BATCH_SIZE,
                                                           shuffle=True)


# In[ ]:


# MobileNet
# URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"#
# feature_extractor = hub.KerasLayer(URL, trainable=False, input_shape=(IMG_SIZE, IMG_SIZE,3))


# In[ ]:


# Inception v3
URL_INC = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL_INC, trainable=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))


# In[ ]:


# Create the model
model = tf.keras.Sequential()
model.add(feature_extractor)
model.add(tf.keras.layers.Dense(2, activation='softmax'))
 
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

history = model.fit_generator(
      train_batches,
      steps_per_epoch=train_batches.samples/train_batches.batch_size ,
      epochs=EPOCHS,
      validation_data=valid_batches,
      validation_steps=valid_batches.samples/valid_batches.batch_size,
      verbose=1)


# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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
plt.savefig('./foo.png')
plt.show()

