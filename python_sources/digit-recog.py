#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Using CNN and tf CPU
"""
import tensorflow as tf
import pandas as pd
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator as ig
from keras.layers import Dropout
import pickle


# In[ ]:



#Making Sequential Object
model = Sequential()


# In[ ]:


#Feature Detection and Pooling
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


#Conversion of image matrix into list of input node
model.add(Flatten())


# In[ ]:


#Hidden Layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


# In[ ]:


#compilation of layres
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


#image augmentation
train_datagen = ig(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ig(rescale=1./255)


# In[ ]:


#Specifying location of train and test data
train_data = train_datagen.flow_from_directory('../input/test/test/trainingSet',
                                                target_size=(28, 28),
                                                batch_size=32,
                                                class_mode='sparse')


test_data = train_datagen.flow_from_directory('../input/test/test/t',
                                                target_size=(28, 28),
                                                batch_size=32,
                                                class_mode='sparse')


# In[ ]:


#Fitting of model
model.fit_generator(
        train_data,
        steps_per_epoch=42000,
        epochs=10,
        validation_data=test_data,
        validation_steps=28000)
#saving the CNN in a pickle string
saved_model = pickle.dumps(model) 


# In[ ]:




