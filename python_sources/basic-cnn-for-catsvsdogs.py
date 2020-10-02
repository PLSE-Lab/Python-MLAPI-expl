#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential


# In[ ]:


#Extract the zip files.
import zipfile
train_zip = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip')
test1_zip = zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip')
train_zip.extractall()
test1_zip.extractall()


# In[ ]:


#Read the images from the train set store the images in one list and labels in another.
train_images = []
labels = []
for image_name in os.listdir('/kaggle/working/train'):
    train_images.append(image_name)
    
    #0 for cats and 1 for dogs.
    if 'cat' in image_name:
        labels.append(str(0))
    else:
        labels.append(str(1))
        
print(len(train_images))
print(train_images[0:5])
print(len(labels))
print(labels[0:5])


# In[ ]:


from sklearn.model_selection import train_test_split
train_dataframe = pd.DataFrame({'file_names' : train_images, 'labels' : labels})
train_dataframe.head()

train_dataframe, validation_dataframe = train_test_split(train_dataframe, test_size = 0.2, shuffle = True)
print(train_dataframe.shape)
print(validation_dataframe.shape)


# In[ ]:


#Data Augmentation.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(    
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255.,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)


# In[ ]:


train_path = '/kaggle/working/train'

#Creating the generators for the train and validation dataset.
train_generator = train_datagen.flow_from_dataframe(
    train_dataframe, 
    train_path, 
    x_col='file_names',
    y_col='labels',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=64
)

valid_generator = valid_datagen.flow_from_dataframe(
    validation_dataframe, 
    train_path, 
    x_col='file_names',
    y_col='labels',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=64
)


# In[ ]:


#Define the structure of the model.
model = tf.keras.models.Sequential([
    # the images were resized by ImageDataGenerator 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # since we have only 2 classes to predict we can use 1 neuron and sigmoid
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99),
    loss='binary_crossentropy',
    metrics = ['accuracy'])


# In[ ]:


history = model.fit_generator(train_generator,
    validation_data=valid_generator,
    epochs=20,
)


# In[ ]:


test_filenames = os.listdir('/kaggle/working/test1')
test_df = pd.DataFrame({
    'id': test_filenames
})

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    '/kaggle/working/test1', 
    x_col='id',
    y_col=None,
    class_mode=None,
    target_size=(150, 150),
    batch_size=64,
    shuffle=False
)

yhat = model.predict_generator(test_generator)


# In[ ]:


yhat = [1 if y > 0.5 else 0 for y in yhat]
test_df['label'] = yhat

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)
test_df.to_csv('submission.csv', index=False)


# In[ ]:


model.save('model.h1')

