#!/usr/bin/env python
# coding: utf-8

# # Aerial Cactus Identification

# In this kernel we will use Convolutional Neural Networks to build a model that can identify if there is a cactus in an image. **If you found this kernel useful then please consider upvoting :)**

# Here we will use [Keras](https://keras.io/) library to create the CNN.
# 
# We will create this kernel in four main steps:
# * Get the data
# * Visualise the data
# * Implement the model
# * Make predictions

# ## Getting Started

# This is the first step in our pipeline. In this step we will import the required libraries and then we will import the training and test data.

# In[ ]:


# Import Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import cv2 
import os

import keras.backend as k
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[ ]:


# Getting the training images and labels
train = pd.read_csv('../input/train.csv')

train_labels = train['has_cactus']
train_images = []

for img in tqdm(train['id']):
    img_path = '../input/train/train/'+img;
    train_images.append(cv2.resize(cv2.imread(img_path), (70, 70)))
train_X = np.asarray(train_images)
train_Y = pd.DataFrame(train_labels)


# ## Visualising the Data

# Now let's visualise images and labels in the dataset to get a little bit of intuition about the data. Here we have displayed an image for each label.

# In[ ]:


plt.title(train_Y['has_cactus'][0])
_ = plt.imshow(train_X[0])


# In[ ]:


plt.title(train_Y['has_cactus'][1000])
_ = plt.imshow(train_X[1000])


# Here, we will split the dataset into training and test set data. We will use the training data for training purposes and test data will help us analyse how well our model works on previously unseen data elements. We have used scikit-learn's [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function for this purpose.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)


# ## Implementing the model

# ### Convolutional Neural Networks

# In this kernel we will be using a CNN implemented with the help of Keras for required task.

# CNNs are very much similar to the traditional Neural Networks. CNNs are specially designed for Image recoginition and Computer Vision purposes. They already assume that the input will be an image and hence allows us to encode them accordingly. For more details on CNNs read [this](http://cs231n.github.io/convolutional-networks/)

# ![](https://www.researchgate.net/publication/323227084/figure/fig3/AS:594709642756096@1518801236681/Structure-of-the-convolutional-neural-network.png)

# In[ ]:


import keras
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:



input_shape = (70, 70, 3)
dropout_dense_layer = 0.6

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# We will use the ImageDataGenerator() for processing our Images. It generates tensors of the image data with real-time data augmentation. For more details on how to use it go [here](https://machinelearningmastery.com/image-augmentation-deep-learning-keras/).

# In[ ]:


datagen = ImageDataGenerator()


# In[ ]:


datagen.fit(x_train)


# In[ ]:


history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=50), steps_per_epoch=x_train.shape[0], epochs=2, validation_data=(x_test, y_test), verbose=1)


# In[ ]:


[loss, accuracy] = model.evaluate(x_test, y_test)


# In[ ]:


print('Test Set Accuracy: '+str(accuracy*100)+"%");


# ## Making Predictions

# Finally, we have got a good accuracy and now it's time to make predictions for the test set data

# In[ ]:


# Getting the test set images
test_path = '../input/test/test/'
test_images_names = []

for filename in tqdm(os.listdir(test_path)):
    test_images_names.append(filename)
    
test_images_names.sort()

images_test = []

for image_id in tqdm(test_images_names):
    images_test.append(np.array(cv2.resize(cv2.imread(test_path + image_id), (70, 70))))
    
images_test = np.asarray(images_test)
images_test = images_test.astype('float32')
images_test /= 255


# In[ ]:


# making predictions
prediction = model.predict(images_test)


# In[ ]:


predict = []
for i in range(len(prediction)):
    if prediction[i][0]>0.5:
        answer = prediction[i][0]
    else:
        answer = prediction[i][0]
    predict.append(answer)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['has_cactus'] = predict


# In[ ]:


# Creating the final submission file
submission.to_csv('sample_submission.csv',index = False)


# In[ ]:




