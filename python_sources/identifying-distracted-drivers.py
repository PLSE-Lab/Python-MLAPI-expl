#!/usr/bin/env python
# coding: utf-8

# # Identifying distracted drivers
# 
# In this notebook, I'll use the dataset which includes images of drivers while performing a number of tasks including .... The aim is to correctly identify if the driver is distracted from driving. We might also like to check what activity is the person performing.

# # Import libraries
# 
# I'll use Keras and Tensorflow libraries to create a **Convolutional Neural Network**. So, I'll import the necessary libraries to do the same.

# In[ ]:


import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 = INFO, WARNING, and ERROR messages are not printed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


# # Import the dataset
# 
# I'll import the .csv file to read the labels.

# In[ ]:


dataset = pd.read_csv('../input/driver_imgs_list.csv')
dataset.head(5)


# From the csv file, I'll use the `classname` as the labels for the images and use the image names to match the labels with the correct images.

# # Images overview
# 
# Let's take a look at the various images in the dataset. I'll plot an image for each of the 10 classes. As the directory names are not descriptive, I'll use a map to define the title for each image that is more descriptive.

# In[ ]:


import os
from IPython.display import display, Image
import matplotlib.image as mpimg

activity_map = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}

plt.figure(figsize = (12, 20))
image_count = 1
BASE_URL = '../input/train/'
for directory in os.listdir(BASE_URL):
    if directory[0] != '.':
        for i, file in enumerate(os.listdir(BASE_URL + directory)):
            if i == 1:
                break
            else:
                fig = plt.subplot(5, 2, image_count)
                image_count += 1
                image = mpimg.imread(BASE_URL + directory + '/' + file)
                plt.imshow(image)
                plt.title(activity_map[directory])


# # Building the model
# 
# I'll develop the model with a total of 3 Convolutional layers, then a Flatten layer and then 3 Dense layers. I'll use the optimizer as `adam`, and loss as `categorical_crossentropy`.

# In[ ]:


classifier = Sequential()
classifier.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', input_shape = (240, 240, 3), data_format = 'channels_last'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()


# # Creating training data
# 
# Once the model is ready, I'll use the data on which I want to train the model. The folder `train` includes the images I need. I'll generate more images using **ImageDataGenerator** and split the training data into 80% train and 20% validation split.

# In[ ]:


train_datagen = ImageDataGenerator(rescale = 1.0/255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True, 
                                   validation_split = 0.2)

training_set = train_datagen.flow_from_directory('../input/train', 
                                                 target_size = (240, 240), 
                                                 batch_size = 32,
                                                 subset = 'training')

validation_set = train_datagen.flow_from_directory('../input/train', 
                                                   target_size = (240, 240), 
                                                   batch_size = 32,
                                                   subset = 'validation')


# # Train the model
# 
# Using `fit_generator`, I'll train the model. I'll also save the model to the file, `safeDriving.h5`.

# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch = 17943/32,
                         epochs = 10,
                         validation_data = validation_set,
                         validation_steps = 4481/32)


# # Predicting on test data

# In[ ]:


from PIL import Image

def get_data(image_path):
    img = Image.open(image_path)
    img = img.resize((240, 240), Image.ANTIALIAS) # resizes image in-place
    return np.asarray(img)/255


# In[ ]:


test_file = pd.read_csv('../input/sample_submission.csv')
test_file.head(5)


# In[ ]:


for i, file in enumerate(test_file['img']):
    image = get_data('../input/test/' + file)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    result = classifier.predict(image)
    test_file.iloc[i, 1:] = result[0]


# In[ ]:


test_file.to_csv('results.csv', index = False)

