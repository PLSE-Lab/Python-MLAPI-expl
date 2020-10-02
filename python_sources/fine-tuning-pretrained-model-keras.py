#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# Build an image tagging Deep Learning model that can help the company classify these images into eight categories of Indian classical dance.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Importing Required Libraries[](http://)

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import os
import random
from shutil import copyfile


# Reading the train and test directory containing images and csv files containing images names and labels

# In[ ]:


train_dir = '/kaggle/input/train/'
test_dir = '/kaggle/input/test/'
train_csv = pd.read_csv(r'/kaggle/input/train.csv')
test_csv = pd.read_csv(r'/kaggle/input/test.csv')


# In[ ]:


train_csv.head()


# ## Preparing Data for training

# Making the required directories for flow_from_directory command
# 
# Making source directory with 8 sub categories named as the 8 dance forms given in the train.csv file to keep all the images present in the train directory provided for the competition
# 
# Similarly, Making training directory with 8 sub categories named as the 8 dance forms given in the train.csv file to keep the images used for training after the splitting of train and validation images
# 
# Similarly, Making testing directory with 8 sub categories named as the 8 dance forms given in the train.csv file to keep the images used for validation after the splitting of train and validation images
# 
# Making tests directory with sub directory unknown to contain test directory images for predict_generator

# In[ ]:


os.mkdir(r'/kaggle/working/identify-dance-form')
os.mkdir(r'/kaggle/working/identify-dance-form/training')
os.mkdir(r'/kaggle/working/identify-dance-form/testing')
os.mkdir(r'/kaggle/working/identify-dance-form/source')
os.mkdir(r'/kaggle/working/identify-dance-form/training/manipuri')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/manipuri')
os.mkdir(r'/kaggle/working/identify-dance-form/source/manipuri')
os.mkdir(r'/kaggle/working/identify-dance-form/training/bharatanatyam')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/bharatanatyam')
os.mkdir(r'/kaggle/working/identify-dance-form/source/bharatanatyam')
os.mkdir(r'/kaggle/working/identify-dance-form/training/odissi')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/odissi')
os.mkdir(r'/kaggle/working/identify-dance-form/source/odissi')
os.mkdir(r'/kaggle/working/identify-dance-form/training/kathakali')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/kathakali')
os.mkdir(r'/kaggle/working/identify-dance-form/source/kathakali')
os.mkdir(r'/kaggle/working/identify-dance-form/training/kathak')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/kathak')
os.mkdir(r'/kaggle/working/identify-dance-form/source/kathak')
os.mkdir(r'/kaggle/working/identify-dance-form/training/sattriya')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/sattriya')
os.mkdir(r'/kaggle/working/identify-dance-form/source/sattriya')
os.mkdir(r'/kaggle/working/identify-dance-form/training/kuchipudi')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/kuchipudi')
os.mkdir(r'/kaggle/working/identify-dance-form/source/kuchipudi')
os.mkdir(r'/kaggle/working/identify-dance-form/training/mohiniyattam')
os.mkdir(r'/kaggle/working/identify-dance-form/testing/mohiniyattam')
os.mkdir(r'/kaggle/working/identify-dance-form/source/mohiniyattam')
os.mkdir('/kaggle/working/identify-dance-form/tests')
os.mkdir('/kaggle/working/identify-dance-form/tests/unknown')


# Reading the jpg files from the train directory and saving their names in the list named files

# In[ ]:


files = []
for  r, d, f in os.walk(train_dir):
    for file in f:
        if '.jpg' in file:
            files.append(file)


# One of the images from train directory

# In[ ]:


import matplotlib.image as mpimg
plt.imshow(mpimg.imread(os.path.join(train_dir,files[0])))


# Copying images from the train directory to the Source directory's sub folders based on the labels of the images provided in train.csv file

# In[ ]:


for x in files:
    if (train_csv[train_csv['Image'] == x]['target'] == 'odissi').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/odissi/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'manipuri').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/manipuri/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'bharatanatyam').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/bharatanatyam/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'kathakali').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kathakali/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'kathak').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kathak/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'sattriya').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/sattriya/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'kuchipudi').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kuchipudi/',x)
        copyfile(train_temp, final_train)
    elif (train_csv[train_csv['Image'] == x]['target'] == 'mohiniyattam').bool():
        train_temp = os.path.join(train_dir,x)
        final_train = os.path.join('/kaggle/working/identify-dance-form/source/mohiniyattam/',x)
        copyfile(train_temp, final_train)


# Defining the function for splitting the Source Diretory images into training and testing(validation) directories 

# In[ ]:


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    shuffle=random.sample(os.listdir(SOURCE),len(os.listdir(SOURCE)))
    train_data_length=int(len(os.listdir(SOURCE))*SPLIT_SIZE)
    test_data_length=int(len(os.listdir(SOURCE))-train_data_length)
    train_data=shuffle[0:train_data_length]
    test_data=shuffle[-test_data_length:]
    for x in train_data:
        train_temp=os.path.join(SOURCE,x)
        final_train=os.path.join(TRAINING,x)
        copyfile(train_temp,final_train)
    for x in test_data:
        test_temp=os.path.join(SOURCE,x)
        final_test=os.path.join(TESTING,x)
        copyfile(test_temp,final_test)


# In[ ]:


bhatanatyam_source_dir = '/kaggle/working/identify-dance-form/source/bharatanatyam/'
bhatanatyam_training_dir = '/kaggle/working/identify-dance-form/training/bharatanatyam/'
bhatanatyam_testing_dir = '/kaggle/working/identify-dance-form/testing/bharatanatyam/'

kathak_source_dir = '/kaggle/working/identify-dance-form/source/kathak/'
kathak_training_dir = '/kaggle/working/identify-dance-form/training/kathak/'
kathak_testing_dir = '/kaggle/working/identify-dance-form/testing/kathak/'

kathakali_source_dir = '/kaggle/working/identify-dance-form/source/kathakali/'
kathakali_training_dir = '/kaggle/working/identify-dance-form/training/kathakali/'
kathakali_testing_dir = '/kaggle/working/identify-dance-form/testing/kathakali/'

kuchipudi_source_dir = '/kaggle/working/identify-dance-form/source/kuchipudi/'
kuchipudi_training_dir = '/kaggle/working/identify-dance-form/training/kuchipudi/'
kuchipudi_testing_dir = '/kaggle/working/identify-dance-form/testing/kuchipudi/'

manipuri_source_dir = '/kaggle/working/identify-dance-form/source/manipuri/'
manipuri_training_dir = '/kaggle/working/identify-dance-form/training/manipuri/'
manipuri_testing_dir = '/kaggle/working/identify-dance-form/testing/manipuri/'

mohiniyattam_source_dir = '/kaggle/working/identify-dance-form/source/mohiniyattam/'
mohiniyattam_training_dir = '/kaggle/working/identify-dance-form/training/mohiniyattam/'
mohiniyattam_testing_dir = '/kaggle/working/identify-dance-form/testing/mohiniyattam/'

odissi_source_dir = '/kaggle/working/identify-dance-form/source/odissi/'
odissi_training_dir = '/kaggle/working/identify-dance-form/training/odissi/'
odissi_testing_dir = '/kaggle/working/identify-dance-form/testing/odissi/'

sattriya_source_dir = '/kaggle/working/identify-dance-form/source/sattriya/'
sattriya_training_dir = '/kaggle/working/identify-dance-form/training/sattriya/'
sattriya_testing_dir = '/kaggle/working/identify-dance-form/testing/sattriya/'


# Splitting the Source Diretory images into training and testing(validation) sub directories 

# In[ ]:


split_size = 0.85
split_data(bhatanatyam_source_dir, bhatanatyam_training_dir, bhatanatyam_testing_dir, split_size)
split_data(sattriya_source_dir, sattriya_training_dir, sattriya_testing_dir, split_size)
split_data(odissi_source_dir, odissi_training_dir, odissi_testing_dir, split_size)
split_data(mohiniyattam_source_dir, mohiniyattam_training_dir, mohiniyattam_testing_dir, split_size)
split_data(manipuri_source_dir, manipuri_training_dir, manipuri_testing_dir, split_size)
split_data(kuchipudi_source_dir, kuchipudi_training_dir, kuchipudi_testing_dir, split_size)
split_data(kathakali_source_dir, kathakali_training_dir, kathakali_testing_dir, split_size)
split_data(kathak_source_dir, kathak_training_dir, kathak_testing_dir, split_size)


# Importing VGG16 Model for Pre-training

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16


# Initializing VGG16 Model

# In[ ]:


pre_trained_model = VGG16(include_top = False,
                            input_shape = (156,156,3),
                            weights = 'imagenet')


# In[ ]:


pre_trained_model.summary()


# Setting the layers of pre trained model to be trainable

# In[ ]:


pre_trained_model.trainable = True

print(len(pre_trained_model.layers))


# ### Fine Tuning the VGG16 Model

# In[ ]:


fine_tune_at = 17


# Making some of the layers non-trainable of the VGG16 Model

# In[ ]:


for layer in pre_trained_model.layers[:fine_tune_at]:
    layer.trainable = False


# In[ ]:


last_output = pre_trained_model.output


# Adding some layers to implement VGG16 Model to fit well on our Dataset

# In[ ]:


x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(8, activation = 'softmax')(x)


# ## Model Formation

# In[ ]:


model = tf.keras.Model(pre_trained_model.input, x)


# In[ ]:


model.summary()


# Initializing ImageDataGenerator and applying **Image Augmentation**

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAINING_DIR = "/kaggle/working/identify-dance-form/training"
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.1,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                   target_size=(156,156),
                                                   color_mode = 'rgb',
                                                   batch_size=32,
                                                   class_mode='categorical')

VALIDATION_DIR = "/kaggle/working/identify-dance-form/testing"
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                   target_size=(156,156),
                                                   color_mode = 'rgb',
                                                   batch_size=32,
                                                   class_mode='categorical')


# Initializing the Model

# In[ ]:


model.compile(tf.keras.optimizers.RMSprop(lr = 0.001), loss='categorical_crossentropy', metrics=['acc'])


# Initializing the Callback

# In[ ]:


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# Fitting the Model

# In[ ]:


history = model.fit_generator(train_generator,
                              epochs=40,
                              verbose=1,
                              validation_data=validation_generator,
                             callbacks = [learning_rate_reduction])


# Graphical Comparison of Training and validation - accuracy and loss

# In[ ]:


plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss') 
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['acc'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_acc'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)


# Reading the jpg files from the test directory and saving their names in the list named files

# ## Predicting the test images labels and preparing submission file

# In[ ]:


fil = []
for  r, d, f in os.walk(test_dir):
    for file in f:
        if '.jpg' in file:
            fil.append(file)


# One of the images from train directory

# In[ ]:


plt.imshow(mpimg.imread(os.path.join(test_dir,fil[0])))


# Copying images from the train directory to the tests directory - to prepare it for predict_generator function

# In[ ]:


for x in fil:
    train_temp = os.path.join(test_dir,x)
    final_train = os.path.join('/kaggle/working/identify-dance-form/tests/unknown',x)
    copyfile(train_temp, final_train)


# In[ ]:


tests_dir = '/kaggle/working/identify-dance-form/tests'
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(tests_dir,
                                                  target_size = (156,156),
                                                  color_mode = 'rgb',
                                                  batch_size=32,
                                                  class_mode=None,
                                                  shuffle=False,
                                                  seed=42)


# Saving train images names in img_list

# In[ ]:


img_list = []
for x in test_generator.filenames:
    x = x.split('/')[1]
    img_list.append(x)


# Making prediction on test images using predict_trainer

# In[ ]:


predictions = model.predict_generator(test_generator)


# Converting the probabalities we got from softmax layers into the integer labels

# In[ ]:


predicted_clases = np.argmax(predictions,axis=-1)


# Class Indices assigned to the sub classes by the train_generator

# In[ ]:


train_generator.class_indices


# Creating a DataFrame with the image name and the predicted image label

# In[ ]:


data = {'Image': img_list, 'target': predicted_clases}


# In[ ]:


df = pd.DataFrame(data)
df.head()


# Mapping back the class indices with the class label name

# In[ ]:


df['target']= df['target'].map({0: 'bharatanatyam',
                                1: 'kathak',
                                2: 'kathakali',
                                3: 'kuchipudi',
                                4: 'manipuri',
                                5: 'mohiniyattam',
                                6: 'odissi',
                                7: 'sattriya'})


# In[ ]:


df.head()


# Exporting csv file for submission

# In[ ]:


df.to_csv(r'submission_dance.csv', index = False)


# # If You like it DO UPVOTE
