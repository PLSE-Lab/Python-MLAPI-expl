#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np
import pandas as pd 
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


base_dir = "../input"


# In[ ]:


train_df = pd.read_csv(base_dir + '/train.csv')


# In[ ]:


train_df.dtypes


# In[ ]:


train_df.head()


# In[ ]:


os.listdir(base_dir)


# In[ ]:


train_images_dir = base_dir + '/train/train'
test_images_dir = base_dir + '/test/test'


# In[ ]:


train_images_dir


# In[ ]:


len(os.listdir(train_images_dir))


# In[ ]:


os.listdir(train_images_dir)[0]


# In[ ]:


images_paths_list = glob(train_images_dir + '/*.jpg')


# In[ ]:


len(images_paths_list)


# In[ ]:


images_paths_list[0]


# In[ ]:


image_id_path_dict = {os.path.basename(x): x for x in images_paths_list}


# In[ ]:


list(image_id_path_dict.keys())[:3]


# In[ ]:


exact_paths_list = [image_id_path_dict[x] for x in list(train_df["id"])]


# In[ ]:


len(exact_paths_list)


# In[ ]:


exact_paths_list[0]


# In[ ]:


train_df["paths"] = exact_paths_list


# In[ ]:


train_df.head(1).values


# In[ ]:


#Get and store images by resizing into 32 x 32
train_df['images'] = train_df['paths'].map(lambda x: np.array(Image.open(x).resize((32,32))))


# In[ ]:


len(train_df)


# In[ ]:


train_df.head(1).values


# In[ ]:


#Normalize the images
train_df['images'] = train_df['images'] / 255


# In[ ]:


train_df["images"][0].shape


# In[ ]:


train_df["has_cactus"].value_counts()


# In[ ]:


type(train_df["has_cactus"].value_counts())


# In[ ]:


# Creating the test df in similar way
test_df = pd.read_csv(base_dir + '/sample_submission.csv')


# In[ ]:


test_df.head()


# In[ ]:


test_images_paths_list = glob(test_images_dir + '/*.jpg')


# In[ ]:


len(test_images_paths_list)


# In[ ]:


test_image_id_path_dict = {os.path.basename(x): x for x in test_images_paths_list}


# In[ ]:


test_exact_paths_list = [test_image_id_path_dict[x] for x in list(test_df["id"])]


# In[ ]:


test_df["paths"] = test_exact_paths_list


# In[ ]:


test_df.head()


# In[ ]:


test_df['images'] = test_df['paths'].map(lambda x: np.array(Image.open(x)))


# In[ ]:


test_df['images'] = test_df['images'] / 255


# In[ ]:


test_df["images"][0].shape


# In[ ]:


#Start modelling

x = train_df['images']
y = train_df['has_cactus']


# In[ ]:


type(x)


# In[ ]:


x = np.array(list(x))
y = np.array(list(y))


# In[ ]:


x.shape,y.shape


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.15, random_state=1234)


# In[ ]:


# Use to reduce learning rate 
lr = ReduceLROnPlateau(monitor= 'val_acc',
                              patience=3,
                              verbose=1,
                              factor=0.5)


# In[ ]:


callbacks = [lr]


# In[ ]:


input_shape = x_train.shape[1:]

model = Sequential([Conv2D(32, kernel_size=(2,2), activation='relu', padding='same', input_shape = input_shape),
                  Conv2D(32, kernel_size=(2,2), activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.2),
                  
                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),
                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.2),
                  
                  Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),
                  Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),
                  MaxPool2D(pool_size=(2,2)),
                  Dropout(0.5),
                  
                  Flatten(),
                  Dense(64, activation='relu'),
                  Dropout(0.5),
                  Dense(28, activation='relu'),
                  Dropout(0.5),
                  Dense(1, activation='sigmoid')
                 ])


# In[ ]:


model.compile(optimizer=Adam(lr=0.001), loss = "binary_crossentropy", metrics=['acc'])


# In[ ]:


#fit the cnn model
model_stats = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_valid,y_valid), callbacks=callbacks)


# In[ ]:


test_pred = model.predict_classes(np.array(list(test_df['images'])))


# In[ ]:


type(test_pred)


# In[ ]:


test_pred.shape


# In[ ]:


test_df['has_cactus'] = np.squeeze(test_pred)


# In[ ]:


test_df.head()


# In[ ]:


test_df.head(1).values


# In[ ]:


submission_df = test_df[['id','has_cactus']]


# In[ ]:


submission_df.to_csv('cactus_detections.csv', index=False)

