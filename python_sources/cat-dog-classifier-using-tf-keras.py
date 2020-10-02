#!/usr/bin/env python
# coding: utf-8

# # A basic CNN Cats and Dogs classifier.

# This basic code is for testing out the Kaggle GPU processing power.

# # Importing some libraries

# In[ ]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


tf.__version__


# # Data Preprocessing

# Preprocessing the Training set

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/dogs-cats-images/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# Preprocessing the Test set

# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../input/dogs-cats-images/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# # Building the CNN

# Initialising the CNN

# In[ ]:


cnn = tf.keras.models.Sequential()


# Step 1 - Convolution

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# Step 2 - Pooling

# In[ ]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Adding a second convolutional layer

# In[ ]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# Step 3 - Flattening

# In[ ]:


cnn.add(tf.keras.layers.Flatten())


# Step 4 - Full Connection

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# Step 5 - Output Layer

# In[ ]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# # Training the CNN

# Compiling the CNN

# In[ ]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Training the CNN on the Training set and evaluating it on the Test set

# In[ ]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# # Making some predictions

# In[ ]:


import numpy as np
from keras.preprocessing import image


# 1. Checking out if model correctly predicts this good boi as dog

# In[ ]:


test_image = image.load_img('../input/single-predictions/cat_or_dog1.jpg')


# In[ ]:


test_image


# In[ ]:


# 1. Reassigning test_image to the same target_size as the other trained images.
# 2. Converting the image to numpy array 
# 3. Adding the batch dimension to the data as while training we had that dimension too.

test_image = image.load_img('../input/single-predictions/cat_or_dog1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


# In[ ]:


result = cnn.predict(test_image)


# In[ ]:


training_set.class_indices # This will show us the indexes of cat and dog


# In[ ]:


if result[0][0] == 1: # First '[0]' represents the batch number. Since there is only 1 batch therefore 0.
                      # Second '[0]' represents the image number. Since there is only 1 image therefore 0.
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)


# Correct ! It's a dog ! 

# 2. Checking if our model predicts kitty cat correctly or not

# In[ ]:


test_image2 = image.load_img('../input/single-predictions/cat_or_dog2.jpg')
test_image2


# In[ ]:


test_image2 = image.load_img('../input/single-predictions/cat_or_dog2.jpg', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)

result = cnn.predict(test_image2)

if result[0][0] == 1: 
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)


# It correctly predicts this as cat too !

# 3. Example -3 

# In[ ]:


test_image3 = image.load_img('../input/single-predictions/cat_or_dog3.jpg')
test_image3


# Just want to point out how cute these pups are !

# In[ ]:


test_image3 = image.load_img('../input/single-predictions/cat_or_dog3.jpg', target_size = (64, 64))
test_image3 = image.img_to_array(test_image3)
test_image3 = np.expand_dims(test_image3, axis = 0)

result = cnn.predict(test_image3)

if result[0][0] == 1: 
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)


# This particluar image had 2 dogs and both of them were kind of out of focus. Our model did well ! 
