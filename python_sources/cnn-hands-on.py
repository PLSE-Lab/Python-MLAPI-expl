#!/usr/bin/env python
# coding: utf-8

# In[7]:


# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[9]:


# Initialising the CNN
classifier = Sequential()
# Step 1  - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3 ,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3 - Flatting
classifier.add(Flatten())
# Step 4 - Full Connection
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


# Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[19]:


training_set = train_datagen.flow_from_directory('../input/dataset/dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
test_set = train_datagen.flow_from_directory('../input/dataset/dataset/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')


# In[ ]:


classifier.fit_generator(training_set, 
                        steps_per_epoch = 8000,
                        epochs = 25,
                        validation_data = test_set,
                        validation_steps = 2000)

