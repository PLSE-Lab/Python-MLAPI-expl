#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[3]:


# initializing cnn
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# Step 1 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 2 - Adding Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# step 2 - Adding pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step 3 - Adding Flattening
classifier.add(Flatten())

# step 4 - Full Connection
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[4]:


# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

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


# In[ ]:


# Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/dataset/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.image_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = clasifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat '

