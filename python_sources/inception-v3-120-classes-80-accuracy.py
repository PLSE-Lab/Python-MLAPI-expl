#!/usr/bin/env python
# coding: utf-8

# Hey Reader , 
# I will be performing following steps :
# 1. Importing the data 
# 2. Data Argumentation And Visualization 
# 3. Importing the INCEPTION V3 ( Transfer learning ) 
# 4. Fully Connected layer 
# 5. Model Training 
# 6. Accuracy And Loss Visualization 
# 
# 
# The following cells are without the output , as kaggle dosen't support the "tf nightly" which is used to import the dataset by the keras generator . 
# I am adding link to my Google Colab notebook , in case you want to see the outputs .
# 
# https://colab.research.google.com/drive/1JxlXsSJqnYTx8ZxEyc_VYGnI4wrHikGT?usp=sharing
# 
# Leave your kind comments for enclusive learning !! 

# # **ABOUT THE DATASET **
# 
# The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in colour and age.
# 

# In[ ]:


input_path= '/input/stanford-dogs-dataset/images/Images'


# # **1. IMPORTING THE LIBRARIES AND DATA **

# In[ ]:


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Loaded all libraries")


# In[ ]:


pip install tf-nightly


# In[ ]:


image_size = (200, 200)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/images/Images",
    validation_split=0.2,
    subset="training",
    label_mode = 'int',
    seed = 1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/images/Images",
    validation_split=0.2,
    subset="validation",
    label_mode = 'int',
    seed =1337,
    image_size=image_size,
    batch_size=batch_size,
)


# **Visualizing the Images **

# In[ ]:



plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# # **2. DATA ARGUMENTATION AND VISUALIZATION **

# In[ ]:


data_augmentation_train = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.Rescaling(scale =1./255),
        layers.experimental.preprocessing.RandomHeight(0.1),
        layers.experimental.preprocessing.RandomWidth(0.1)
     
    ]
)


# In[ ]:


data_augmentation_test = keras.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(scale =1./255)
     
    ]
)


# In[ ]:


augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation_train(x, training=True), y))

augmented_val_ds = val_ds.map(
  lambda x, y: (data_augmentation_test(x, training=True), y))


# In[ ]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation_train(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


# # **3. MODEL PREPARATION **

# # **3.1 IMPORTING THE INCEPTION V3**

# In[ ]:


#Inception V3 

from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape=(200,200,3),
                                               include_top=False,
                                               weights='imagenet')

for layer in pre_trained_model.layers:
  layer.trainable = False

pre_trained_model.summary()


# # **3.2 FULLY CONNECTED LAYER**

# In[ ]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import regularizers

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
Dl_1 = tf.keras.layers.Dropout(rate = 0.2)
#pre_prediction_layer = tf.keras.layers.Dense(240, activation='tanh')
#Dl_2 = tf.keras.layers.Dropout(rate = 0.2)
prediction_layer = tf.keras.layers.Dense(120,activation='softmax')

#Add dropout Layer
model_V3 = tf.keras.Sequential([
  pre_trained_model,
  global_average_layer,
  Dl_1,
  #pre_prediction_layer,
  #Dl_2,
  prediction_layer
])

model_V3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_V3.summary()

# Callbacks

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=2, mode='max')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')


# # **4. TRAINING THE MODEL**

# In[ ]:


hist = model_V3.fit(
           augmented_train_ds.repeat(), steps_per_epoch=int(8000/batch_size), 
           epochs=30, validation_data=augmented_val_ds.repeat(), 
           validation_steps=int(2000/batch_size) , callbacks=[lr_reduce])


# # **5. LOSS AND ACCURACY VISUALIZATION **

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(hist.history[met])
    ax[i].plot(hist.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

