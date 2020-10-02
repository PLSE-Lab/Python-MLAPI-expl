#!/usr/bin/env python
# coding: utf-8

# # HackerEarth Deep Learning challenge: Identify the dance form
# This International Dance Day, an event management company organized an evening of Indian classical dance performances to celebrate the rich, eloquent, and elegant art of dance. Post the event, the company planned to create a microsite to promote and raise awareness among the public about these dance forms. However, identifying them from images is a tough nut to crack.
# You have been appointed as a Machine Learning Engineer for this project. Build an image tagging Deep Learning model that can help the company classify these images into eight categories of Indian classical dance.
# 
# ### Dataset
# The dataset consists of 364 images belonging to 8 categories, namely manipuri, bharatanatyam, odissi, kathakali, kathak, sattriya, kuchipudi, and mohiniyattam.
# The benefits of practicing this problem by using Machine Learning/Deep Learning techniques are as follows:
# This challenge will encourage you to apply your Machine Learning skills to build models that classify images into multiple categories
# This challenge will help you enhance your knowledge of classification actively. It is one of the basic building blocks of Machine Learning and Deep Learning
# We challenge you to build a model that auto-tags images and classifies them into various categories of Indian classical dance forms.
# 
# The data folder consists of two folders and two .csv files. The details are as follows:
# train: Contains 364 images for 8 classes 
# * manipuri,
# * bharatanatyam
# * odissi
# * kathakali
# * kathak
# * sattriya
# * kuchipudi
# * mohiniyattam
# 
# test: Contains 156 images
# train.csv: 364 x 2
# test.csv: 156 x 1
# 
# Data description
# This data set consists of the following two columns:
# 
# | Column Name | Description |
# |-------------|-------------|
# | Image       | Name of Image| 
# |target       |Category of Image  ['manipuri','bharatanatyam','odissi','kathakali','kathak','sattriya','kuchipudi','mohiniyattam'] |

# In this file we are using Transfer Learning concept to classify Indian dance form. Transfer Learning used when we have very less training data. In image processing, training with less data does not give good results. So we are using Transfer Learning to get weights. This notebook use *tensorflow* *VGG16* and the purpose of this notbook to show how to use Transfer Learning for image classification

# In[ ]:


# import required libraries 
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten,Dense,Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load train and test csv file for image class
train = pd.read_csv('/kaggle/input/identifythedanceform/train.csv')
test = pd.read_csv('/kaggle/input/identifythedanceform/test.csv')

print(train.head())
print(test.head())
print(train['target'].value_counts())


# In[ ]:


train.head()


# Basic Histrogram plot to check number of training data for each dance form.

# In[ ]:


#Histogram chart for target
train['target'].value_counts().plot(kind='bar')


# In[ ]:


base='/kaggle/input/identifythedanceform'
train_dir = os.path.join(str(base)+ '/train/')
test_dir = os.path.join(str(base)+'/test/')

train_fnames = os.listdir(train_dir)
test_fnames = os.listdir(test_dir)

print(train_fnames[:9])
print(test_fnames[:9])


# In[ ]:


# Images might be in different size. In this section I assigning all image at same size of 224*224
img_width = 224
img_height = 224


# Below two section used for data preprocessing. We are reading image data using OpenCV and converting into numeric formate.

# In[ ]:


# this function reads image from the disk,train file for image and class maping and returning output in numpy array formate
# for input and target data
def train_data_preparation(list_of_images, train, train_dir):
    """
    Returns two arrays: 
        train_data is an array of resized images
        train_label is an array of labels
    """
    train_data = [] 
    train_label = [] 
    for image in list_of_images:
        train_data.append(cv2.resize(cv2.imread(train_dir+image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
        if image in list(train['Image']):
            train_label.append(train.loc[train['Image'] == image, 'target'].values[0])
    
            
    return train_data, train_label


# In[ ]:


def test_data_prepare(list_of_images, test_dir):
    """
    Returns: 
        x is an array of resized images
    """
    test_data = [] 
    
    for image in list_of_images:
        test_data.append(cv2.resize(cv2.imread(test_dir+image), (img_width,img_height), interpolation=cv2.INTER_CUBIC)) 
            
    return test_data


# In[ ]:


training_data, training_labels = train_data_preparation(train_fnames, train, train_dir)


# In[ ]:


training_labels[:10]


# In[ ]:


training_data[1]


# In[ ]:



def show_batch(image_batch, label_batch):
    plt.figure(figsize=(12,12))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n].title())
        plt.axis('off')


# Just showing loaded data for first 25 image

# In[ ]:


show_batch(training_data, training_labels)


# In[ ]:


testing_data = test_data_prepare(test_fnames, test_dir)


# Using label incoder converting target class to numeric format

# In[ ]:


le =LabelEncoder()
training_labels=le.fit_transform(training_labels)


# In this section I am using ougumentation techniques to generate more data for given input

# In[ ]:


training_labels[:10]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(training_data, training_labels, test_size=0.33, random_state=42)


# In[ ]:


train_datagenerator = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        rotation_range=40,  
        zoom_range = 0.20,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=True,  
        vertical_flip=False) 


val_datagenerator=ImageDataGenerator(
        rescale=1. / 255
)

train_datagenerator.fit(X_train)
val_datagenerator.fit(X_val)
X_train=np.array(X_train)
X_val=np.array(X_val)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# In below code we are loading *VGG16* weights for image classifier using transfer learning

# In[ ]:


# traing using transfer learning

vggmodel =VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3),pooling='max')

 # Print the model summary
vggmodel.summary()


# Using already trained model for our task and bulding 2 fully connected layer with *softmax* activation function

# In[ ]:


vggmodel.trainable = False
model = Sequential([
  vggmodel, 
  Dense(1024, activation='relu'),
  Dropout(0.15),
  Dense(256, activation='relu'),
  Dropout(0.15),
  Dense(8, activation='softmax'),
])


# In[ ]:



reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)

callbacks = [reduce_learning_rate]


# In the below code we are compiling and traing our image data

# In[ ]:


model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history =model.fit_generator(
    train_datagenerator.flow(X_train, to_categorical(y_train,8), batch_size=16),
    validation_data=val_datagenerator.flow(X_val, to_categorical(y_val,8), batch_size=16),
    verbose=2,
    epochs=30,
    callbacks=callbacks
)


# In[ ]:


history.history['val_accuracy']


# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

acc      = history.history['accuracy']
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[ 'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot( epochs, acc )
plt.plot( epochs, val_acc )
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )

