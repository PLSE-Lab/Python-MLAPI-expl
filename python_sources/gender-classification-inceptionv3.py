#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# # Using Pre_trained Model InceptionV3

# In[ ]:


pre_trained_model = InceptionV3(include_top = False,
                               input_shape = (150,150,3),
                               weights='imagenet')


# In[ ]:


for layers in pre_trained_model.layers:
    layers.trainable = False


# In[ ]:


last_layer = pre_trained_model.get_layer('mixed8')
last_output = last_layer.output


# In[ ]:


x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1,activation = 'sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input , x)


# Using pretrained model InceptionV3 for this task.
# Since we have a very short amount of data we have to use a model which is already trained for similar task
# 1- Setting all the layers layers.trainable = False so that we don't have to train it again. 
# 2- Taking much more details so i choose 'mixed8' layer so after that we introduce our own layers to predict the gender.
# 3- first use flatten then DNN of 1024 neuron followed by a dorpout of 20% and then we have a sigmoid actiation function for male and female.
# 4- Now combine the both model pretrained upto mixed8 layer and our 4 layers.

# In[ ]:


hiss = model.compile(optimizer = 'Adam',loss = tf.keras.losses.BinaryCrossentropy(),metrics = ['accuracy'])


# # Data Augmentation
# 
# We first rescale the image and used shear,horizontal flip , vertical flip and zoom on the images as we have very less amount of data.
# saving 10% for the validation .

# In[ ]:


data_gen = ImageDataGenerator(rescale = 1./255 ,
                              width_shift_range = 0.2 ,
                              validation_split=0.1,
                              height_shift_range = 0.2 ,
                              shear_range = 0.2 ,
                              horizontal_flip = True ,
                              vertical_flip = True,
                              zoom_range = 0.2)


# # Training and Validation Data

# In[ ]:


training_data = data_gen.flow_from_directory('/kaggle/input/men-women-classification/data',
                                            target_size = (150,150),
                                            class_mode='binary',
                                            batch_size = 32,
                                            subset = 'training'
                                            )
validation_data = data_gen.flow_from_directory('/kaggle/input/men-women-classification/data',
                                              target_size = (150,150),
                                              class_mode='binary',
                                              batch_size = 32,
                                              subset = 'validation')


# # Training

# In[ ]:


history = model.fit_generator(training_data,epochs = 30 , steps_per_epoch =  93,validation_data = validation_data)


# # Plotting Accuracy 

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# # Plotting Losses

# In[ ]:


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()


# I have written this kernel in very understandable manner. No fancy tricks and demonstrate every possible tricks that you can do to improve the accuracy including Data Augmentation, Tranfer learning and used keras api in order so that even a new data scientist can read it and get most from it. We get an accuracy of 90% which is great.
# If you found any error please report down in the comment section. Upvote the kernel and don't forget to follow me. Cheers!! kagglers.
