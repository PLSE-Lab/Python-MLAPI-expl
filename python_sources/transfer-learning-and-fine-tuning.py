#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# To build a CNN model which can identify various dance form images and label their classes out of 8.
# ## Steps:
# 1. Importing libraries
# 2. Loading Data
# 3. Create Directories and Sub directories
# 4. Splitting Training and Validation set
# 5. Define Model
#     * Loading Pretrained Model
#     * Freeze/Lock layers at end
#     * Add new layer to get required number of output    
# 6. Data Augmentation
# 7. Training Model
# 8. Prediction

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


# ## Import Required Libraries

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

from keras import layers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ### Loading data

# In[ ]:


train_dir = '../input/identifythedanceform/train'
test_dir = '../input/identifythedanceform/test'
train_csv = pd.read_csv('../input/identifythedanceform/train.csv')
test_csv = pd.read_csv('../input/identifythedanceform/test.csv')


# In[ ]:


train_csv.head()


# ## Create directories and sub-directories
# 1. Directories and sub-directories are created to store images. Training directory is created to store training images      and it contains 8 sub-directories based on the names of various dance forms. Same thing is done for validation directory and sub-directories
# 2. Source directory containing sub-directories has been created to store all the images at the initial step and then the images are splitted to training and validation directories repectively. Test directory has been created for the final prediction of images after the model is trained already. 

# In[ ]:


os.mkdir('/kaggle/working/identify-dance-form/')
os.mkdir('/kaggle/working/identify-dance-form/training/')
os.mkdir('/kaggle/working/identify-dance-form/testing/')
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


# ## Extracting file names
# Files names from train_csv are extracted out and saved in a variable in the form of list

# In[ ]:


files = []
for r, d, f in os.walk(train_dir):
    for file in f:
        if '.jpg' in file:
            files.append(file)


# In[ ]:


print('Total images in train csv: ',len(files))
print(files)


# ## Let's plot an image from  training data

# In[ ]:


#Using matplotlib's imperative-style plotting interface
import matplotlib.image as mpimg
plt.imshow(mpimg.imread(os.path.join(train_dir, files[10])))


# ### Copying images from train directory to the source sub-directories according to their labels

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


# ### Helper Function to split source directory images to training and validation directories

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


# Assigning variables to vaious directries

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


# ### Splitting the Source Diretory images into training and validation sub directories

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


# ## Define Model(Transfer Learning)

# VGG-19 is a convolutional neural network which is 19 layers deep. It's pretrained version of the network has trained on more than a million images from the ImageNet database. This network is trained on images with an input size of 224-by-224. By adding few more layers to this pretrained model, we can use it for classification of the various dance forms. As, pretrained model has already extracted out the edges and important features from millions of images, so it can classify images with much higher accuracy.

# In[ ]:


from keras.applications.vgg19 import VGG19


# In[ ]:


# Initializing pretrained model
image_size = [224, 224]

pretrained_model = VGG19(input_shape=image_size + [3], weights = 'imagenet', include_top = False)


# In[ ]:


pretrained_model.summary()


# In[ ]:


len(pretrained_model.layers)


# ### Freeze/lock some of the layers at the end.

# In[ ]:


freeze_layers = 21
for layer in pretrained_model.layers[:freeze_layers]:
    layer.trainable=False


# As per the objective of identifying the dance forms from the images, few layers are added to get 8 outputs using softmax activation function.

# In[ ]:


x = Flatten()(pretrained_model.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.40)(x)
x = Dense(8, activation = 'softmax')(x)
model = Model(inputs=pretrained_model.input, outputs = x)
model.summary()


# ## Data Augmentation
# 1. Image Data Generator changes the differnt size images to the input size.
# 2. Normalize images while loading them.
# 3. Create augmented images in the memory.
# 4. Label them based on their directory names

# In[ ]:


TRAINING_DIR = '/kaggle/working/identify-dance-form/training/'
train_datagen = ImageDataGenerator(rescale = 1/255,
                                  rotation_range =20,
                                  width_shift_range=0.3,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.3,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                   target_size=(224,224),
                                                   color_mode='rgb',
                                                   batch_size=8,
                                                   class_mode='categorical')

VALIDATION_DIR = '/kaggle/working/identify-dance-form/testing/'
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=8,
                                                             class_mode='categorical')


# In[ ]:


# Set optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# Set a Learning Rate Annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           factor=0.5,
                                           min_lr=0.00001)


# In[ ]:


# Training model
history = model.fit_generator(train_generator,
                              epochs=40,
                              verbose=2,
                              validation_data=validation_generator,
                              callbacks= [learning_rate_reduction])


# ### Plot Training & Validation Accuracy

# In[ ]:


plt.figure(figsize=(15,7))
ax1 = plt.subplot(1,2,1)
ax1.plot(history.history['loss'], color='b', label='Training Loss')
ax1.plot(history.history['val_loss'], color='r', label = 'Validation Loss',axes=ax1)
legend = ax1.legend(loc='best', shadow=True)
ax2 = plt.subplot(1,2,2)
ax2.plot(history.history['accuracy'], color='b', label='Training Accuracy') 
ax2.plot(history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
legend = ax2.legend(loc='best', shadow=True)


# ### Predicting the test images labels and preparing submission file

# In[ ]:


test_files = []
for r, d, f in os.walk(test_dir):
    for file in f:
        if '.jpg' in file:
            test_files.append(file)


# In[ ]:


print(test_files)


# ## Plotting test image

# In[ ]:


plt.imshow(mpimg.imread(os.path.join(test_dir,test_files[0])))


# ### Copying images from test input data to unknown subdirectory inside test directory
# As the labels of test images are not known to us, so storing them in a subdirectory names as "unknown"

# In[ ]:


for x in test_files:
    test_temp = os.path.join(test_dir, x)
    final_test = os.path.join('/kaggle/working/identify-dance-form/tests/unknown/', x)
    copyfile(test_temp, final_test)


# In[ ]:


test_dir = '/kaggle/working/identify-dance-form/tests'
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(224, 224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode=None,
                                                 shuffle=False,
                                                 seed=42)


# #### Saving test images in a list

# In[ ]:


img_list = []
for x in test_generator.filenames:
    x = x.split('/')[1]
    img_list.append(x)


# In[ ]:


print(img_list)


# ## Prediction of images

# In[ ]:


prediction = model.predict_generator(test_generator)
prediction = np.argmax(prediction, axis=1)
prediction


# In[ ]:


train_generator.class_indices


# #### Creating a DataFrame with the image name and the predicted image label

# In[ ]:


data = {'Image': img_list, 'target': prediction}
df = pd.DataFrame(data)
df.head()


# In[ ]:


df['target'] = df['target'].map({0:'bharatanatyam',
                                 1:'kathak',
                                 2:'kathakali',
                                 3:'kuchipudi',
                                 4:'manipuri',
                                 5:'mohiniyattam',
                                 6:'odissi',
                                 7:'sattriya'})


# In[ ]:


df.head()


# In[ ]:


# Saving csv
df.to_csv('identifydanceform_submission15.csv', index=False)


# ## Give it a try:)
# ## Upvote if you like this notebook 
