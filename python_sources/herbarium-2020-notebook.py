#!/usr/bin/env python
# coding: utf-8

# Firstly, I'd like to say that I'm novice in machine learning and don't know how this problem can be resolved. However I'm just curious how to work with such amount of data and if I can create something working. Here I'm going to build ML model using Keras.
# 
# Just to save your time. I have managed to create model which achieved 0.0022% of accuracy after 8 hours of learning.

# # Env initialization
# 
# Here standart env initialization plus extra package imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import json
import seaborn as sns

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# All parameters in one place:

# In[ ]:


train_images_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/'
test_images_dir = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/'

train_metadata_file_path = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/train/metadata.json'
test_metadata_file_path = '/kaggle/input/herbarium-2020-fgvc7/nybg2020/test/metadata.json'

num_classes = 32093 + 1
batch_size = 16

steps_per_epoch = int(num_classes / batch_size)

img_height = 1000
img_width = 661

epochs_num = 5


# # Data preprocessing
# First of all we have to look inside metadata files located in `train` and `test` subdirectories.
# 
# ## train/metadata.json
# Let's find out what's in that file. For some reason loading data from the file causes problems. We can avoid them by specifyin encoding explicitly and setting `errors='ignore'` option.

# In[ ]:


with open(train_metadata_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    train_metadata_json = json.load(f)


# After loading data in memory we can have a look on it.

# In[ ]:


#Let's see presented keys
train_metadata_json.keys()


# There are no interesting information under 'info', 'licenses' keys. I'm going to create separate DataFrames for the rest keys in metadata file and merge them together nto one DataFrame.

# In[ ]:


#Create Pandas DataFrame per each data type
train_metadata = pd.DataFrame(train_metadata_json['annotations'])

train_categories = pd.DataFrame(train_metadata_json['categories'])
train_categories.columns = ['family', 'genus', 'category_id', 'category_name']

train_images = pd.DataFrame(train_metadata_json['images'])
train_images.columns = ['file_name', 'height', 'image_id', 'license', 'width']

train_regions = pd.DataFrame(train_metadata_json['regions'])
train_regions.columns = ['region_id', 'region_name']

#Combine DataFrames
train_data = train_metadata.merge(train_categories, on='category_id', how='outer')
train_data = train_data.merge(train_images, on='image_id', how='outer')
train_data = train_data.merge(train_regions, on='region_id', how='outer')

#Remove NaN values
train_data = train_data.dropna()

# Update data types
train_data = train_data.astype({'category_id': 'int32',
                                'id': 'int32',
                                'image_id': 'int32',
                                'region_id': 'int32',
                                'height': 'int32',
                                'license': 'int32',
                                'width': 'int32'})

train_data.info()

#Save DataFrame for future usage.
train_data.to_csv('train_data.csv', index=False)


# In[ ]:


del train_categories
del train_images
del train_regions


# In[ ]:


train_data.head()


# ## test/metadata.json
# 
# Let's do the same with test/metadata.json file

# In[ ]:


with open(test_metadata_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    test_metadata_json = json.load(f)


# In[ ]:


test_metadata_json.keys()


# Test metadata file contains only three entries:
# * images
# * info
# * licenses
# 
# That way we're interested only in `images`. Let's create DataFrame for it.

# In[ ]:


test_data = pd.DataFrame(test_metadata_json['images'])

test_data = test_data.astype({'height': 'int32',
                              'id': 'int32',
                              'license': 'int32',
                              'width': 'int32'})

test_data.to_csv('test_data.csv', index=False)


# # Generators

# Dataset contain lots of images and quite heavy. It's impossible to load all images into memory for model fitting. As a workaround I'll try to use ImageDataGenerator from keras. It loads data 'on the fly' and shouldn't consumes a lot of memory.

# In[ ]:


datagen_without_augmentation = ImageDataGenerator(rescale=1./255)
datagen_with_augmentation = ImageDataGenerator(rescale=1./255, 
                                               featurewise_center=False,
                                               samplewise_center=False,
                                               featurewise_std_normalization=False,
                                               samplewise_std_normalization=False,
                                               zca_whitening=False,
                                               rotation_range = 10,
                                               zoom_range = 0.1,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               horizontal_flip=True,
                                               vertical_flip=False)

train_datagen = datagen_with_augmentation.flow_from_dataframe(dataframe=train_data, 
                                                                 directory=train_images_dir, 
                                                                 x_col='file_name', 
                                                                 y_col='category_id',
                                                                 class_mode="raw",
                                                                 batch_size=batch_size,
                                                                 color_mode = 'rgb',
                                                                 target_size=(img_height,img_width)
                                                             )

val_datagen = datagen_without_augmentation.flow_from_dataframe(dataframe=train_data, 
                                                                 directory=train_images_dir, 
                                                                 x_col='file_name', 
                                                                 y_col='category_id',
                                                                 class_mode="raw",
                                                                 batch_size=batch_size,
                                                                 color_mode = 'rgb',
                                                                 target_size=(img_height,img_width))

#test_datagen = datagen_without_augmentation.flow_from_dataframe(dataframe=test_data,
#                                                               directory=test_images_dir,
#                                                               x_col='file_name',
#                                                               color_mode = 'rgb',
#                                                               class_mode=None,
#                                                               target_size=(img_height,img_width))


# Another problem I came across is that Keras expects targets as 'one hot' encoded vectors. But there are 32093 different categories. Storing such sparse vectors for more than 1 million images... takes too much memory. Unfortunately, I wasn't able to find any existing setting for solving that problem. I decided to wrap up ImageDataGenerator inside my own generator. That way I can retrieve portion of data posthandle it and yield as ImageDataGenerator does.

# In[ ]:


def generator_wrapper(generator, num_of_classes):
    for (X_vals, y_vals) in generator:
        Y_categorical = to_categorical(y_vals, num_classes=num_of_classes)
        
        yield (X_vals, Y_categorical)        
        
train_datagen_wrapper = generator_wrapper(train_datagen, num_classes)
val_datagen_wrapper = generator_wrapper(val_datagen, num_classes)


# # Model

# I wasn't going to create model which can perform well on such amount of data just because I don't have enough computation resources. So here very simple model just to check that learning works.

# In[ ]:


model = Sequential()

model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(img_height, img_width, 3), padding='Same', strides=2))
model.add(Conv2D(64, kernel_size=5, activation='relu', padding='Same', strides=2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='Same', strides=2))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='Same', strides=2))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes / 100))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

optimizer = RMSprop(lr=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


import time

start = time.time()

# history = model.fit_generator(train_datagen_wrapper, 
#                               epochs=epochs_num, 
#                               validation_data=val_datagen_wrapper, 
#                               steps_per_epoch=steps_per_epoch, 
#                               validation_steps=steps_per_epoch)
# 

end = time.time()

print(f"\nLearning took {end - start}")


# After 5 epochs and almost 8 hours of learning model reached 0.0022% of accuracy. Not so good but not as bad for such simple model and 32k different classes. Hope this might be helpful.
