#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ## Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
from PIL import Image
import random

import cv2   # object detection
import tensorflow as tf  # deeplearning library
from tensorflow import keras
from sklearn.model_selection import train_test_split  # splitting my nn data easily
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



import sys
import os
print(os.listdir("../input"))
print(os.listdir("../input/natural-images/data/natural_images"))


from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

sys.path.append("..")


# In[ ]:


#from imageai.Detection import ObjectDetection
print(tf.__version__)


# # Installs

# ## Env Setup

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Global Vars

# In[ ]:


inp_img_width  = 224 #200
inp_img_height = 224 #100
img_size = inp_img_height * inp_img_width


# ## Define the Image-Sets

# In[ ]:


def load_images(image_dir):
    """Loads all images inside the imageDir into an array."""
    image_bond = [] # array for all images  
    image_path_list = []
    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"] # valid extensions

    for file in os.listdir(image_dir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        image_path_list.append(os.path.join(image_dir, file))
        
    for imagePath in image_path_list:       
        img = cv2.imread(imagePath)  # reads all the images from the given path
        if img is None: # choose next
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_bond.append(img)
        
    return(image_bond)


def resize_images(img_arr, width, height):
    """Resizes a single image""" 
    img_res_arr = []
    width = width
    height = height
    
    for img in img_arr:
        img = cv2.resize(img,(width, height))
        img_res_arr.append(img)
    
    return img_res_arr

def gray_images(img_arr):
    img_arr_gray = []
    for img in img_arr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_arr_gray.append(img)
    return img_arr_gray

def assign_images(img_arr,assignment):
    #img_arr_assigned = []
    #assignment_arr = [assignment]
    img_arr_assigned = [assignment for img in img_arr]
    #for img in img_arr:
    #    img_arr_assigned.append(assignment)
    return img_arr_assigned


# The following part will load, resize and gray the image for the neural network. Because there are no predefined targets for the images I need to create them by myself based on the image foldername.

# In[ ]:


## Load images, resize and gray them and at least assign the correct object type

## Airplanes
images_airplanes = load_images('../input/natural-images/data/natural_images/airplane/')          ## Load image into array of arrays
images_airplanes = resize_images(images_airplanes, inp_img_width,inp_img_height)  ## Resize the image to the predefined size
#images_airplanes = gray_images(images_airplanes)                                  ## Gray images 
images_airplanes_assigned = assign_images(images_airplanes,'airplane')            ## Assign the correct object to the image (our target)

## Motorbike
images_motorbikes = load_images('../input/natural-images/data/natural_images/motorbike/')
images_motorbikes = resize_images(images_motorbikes, inp_img_width,inp_img_height)
#images_motorbikes = gray_images(images_motorbikes)
images_motorbikes_assigned = assign_images(images_motorbikes,'motorbike')

# Cars
images_cars      = load_images('../input/natural-images/data/natural_images/car/')
images_cars      = resize_images(images_cars, inp_img_width,inp_img_height)
#images_cars      = gray_images(images_cars)
images_cars_assigned = assign_images(images_cars,'car')

# Cats
images_cats       = load_images('../input/natural-images/data/natural_images/cat/')
images_cats       = resize_images(images_cats, inp_img_width,inp_img_height)
#images_cats       = gray_images(images_cats)
images_cats_assigned = assign_images(images_cats,'cat')

# Persons
images_persons    = load_images('../input/natural-images/data/natural_images/person/')
images_persons    = resize_images(images_persons, inp_img_width,inp_img_height)
#images_persons    = gray_images(images_persons)
images_persons_assigned = assign_images(images_persons,'person')


# Next step will be the combination of all the lists above into two lists:
# 
# * features and
# * targets

# #### Features

# In[ ]:


images_features = images_airplanes + images_motorbikes + images_cars + images_cats + images_persons
print('lenght of feature set: ',len(images_features))


# #### Targets

# In[ ]:


images_targets = images_airplanes_assigned + images_motorbikes_assigned + images_cars_assigned + images_cats_assigned + images_persons_assigned
print('lenght of target set: ',len(images_targets))


# ### Random Shuffle of the two Lists
# I will use here a random shuffle for this two lists in a combined way to make sure all targets fit to their features and vice versa. It would be easier to hold both values (array-list based features and the targets) in one list instead of in two ones, but according to the array-list based features, it was difficult for me to find a good version that did not provoke the performances of this notebook. If you have any suggestions, please let me know in the comments.

# In[ ]:


# combine them in one list
comb_list = list(zip(images_features, images_targets))

# shuffle both list equaly
random.Random(45).shuffle(comb_list)

# splitt them again
images_features, images_targets = zip(*comb_list)


# Here the test now that feature (image) and target (typ) map.

# In[ ]:


from IPython.display import display

display(Image.fromarray(images_features[0]))
display(images_targets[0])


# In[ ]:


display(Image.fromarray(images_features[1]))
display(images_targets[1])


# In[ ]:


display(Image.fromarray(images_features[10]))
display(images_targets[10])


# # The Model

# ## Preparing Training
# In this chapter I will prepare the training set which will be a training set from different images of vehicles, persons and animals.

# In[ ]:


# splitting features and targets into train- and test- set
X_train, X_test, y_train, y_test = train_test_split(images_features, images_targets, test_size = 0.20, random_state = 45)


# #### Features

# In[ ]:


# scaling, normalizing the image pixels (between 0 and 1)
X_train = tf.keras.utils.normalize(np.asfarray(X_train))#, axis = -1) 
X_test = tf.keras.utils.normalize(np.asfarray(X_test))#, axis = -1)


# In[ ]:


#X_train = np.asfarray(X_train)
#X_test = np.asfarray(X_test)


# #### Targets

# In[ ]:


# join train and test to encode(get_dummies) all categories in the same way 
y = y_train + y_test
df_y = pd.DataFrame(y) 


# In[ ]:


# Label encoding of the targets
le = LabelEncoder()
le.fit(df_y)
df_y_hot = le.transform(df_y)


# In[ ]:


# reshaping for the neural network
df_y_hot = pd.DataFrame(df_y_hot)
df_y_hot = np.asfarray(df_y_hot)


# In[ ]:


# write the created dummies back again
# just to make this clear: I used the length of the train data set to split the combined lists into train ([:len(y_train)]) and test([len(y_train):]) again.
#  in the same way I convert the result into in array, which is necessary for the use of an nn.
y_train = np.asfarray( df_y_hot[:len(y_train)] )#.reshape(1,-5)
y_test = np.asfarray( df_y_hot[len(y_train):] )#.reshape(1,-5)


# ## Model Building and Training with TensorFlow

# In[ ]:


#y_train.reshape(1,-1)
y_train.shape


# In[ ]:


#y_train[0
X_train[0].shape

#X_train[0]
#Image.fromarray(np.asfarray(X_train[0]))

#display(Image.fromarray(images_features[0]))


# In[ ]:


# shape of the feauture set
print("rows, height, length, color shape : ", X_train.shape)


# In[ ]:


# Model building
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(224, (2,2), activation=tf.nn.relu, input_shape=(224,224,3)))  
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(112, (3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(224, (3,3), activation=tf.nn.relu))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

##model.add(tf.keras.layers.Flatten())


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
             )

model.fit(X_train, y_train, epochs = 4)


# ### Validation Loss
# 

# In[ ]:


# Test the model and show the accuracy
val_loss, val_acc = model.evaluate(X_test, y_test)


# In[ ]:


model.save('first_fit_4e.model')


# In[ ]:





# In[ ]:




