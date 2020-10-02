#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
import random
import time
import cv2

from skimage import io
from pylab import rcParams

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import data, color

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift,img_to_array, ImageDataGenerator)

import numpy as np
import pandas as pd

import warnings
from glob import glob

print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)


# In[ ]:


TRAIN_IMAGE_PATH = "../input/train/"
TEST_IMAGE_PATH = "../input/test/"
TRAINING_DATA='../input/train.csv'
IMG_SIZE = 64


# In[ ]:


df_train = pd.read_csv(TRAINING_DATA)
#print(df_train.iloc[0]['Id'])
#df_train.index.name = 'index'
#df_train = df_train.query('index < 10000')
del_ind = []

for i in range(len(df_train)):
    if df_train.iloc[i]['Id'] == 'new_whale':
        del_ind.append(i)
df_train = df_train.drop(df_train.index[del_ind])
print(len(df_train))

#string to unique int
#set unique int value for each unique classes sring.. string to int
unique_calsses_value = np.unique(df_train[['Id']].values)
#unique_calsses_value = np.delete(unique_calsses_value,[0])
print(unique_calsses_value)
unique_classes_id_dict = {}
unique_id_classes_dict = {}
for i in range(len(unique_calsses_value)):
    unique_classes_id_dict[unique_calsses_value[i]] = i
    unique_id_classes_dict[i] = unique_calsses_value[i]
print(len(unique_id_classes_dict))


# In[ ]:


df_train['classes_id'] = df_train.apply (lambda row: unique_classes_id_dict.get(row['Id']),axis=1)

#df_train = df_train.dropna()
df_train.head(15)
len(df_train)


# In[ ]:


def show_image(image):
    plt.imshow(image)
def plot_images(images):
    rcParams['figure.figsize'] = 14, 8
    plt.gray()
    fig = plt.figure()
    for i in range(min(9, images.shape[0])):
        fig.add_subplot(3, 3, i+1)
        show_image(images[i])
    plt.show()   


# In[ ]:


#resize the image
def LoadImage(img_path):
    image = color.rgb2gray(io.imread(img_path))
    image_resized = resize(image,(IMG_SIZE,IMG_SIZE))
    return image_resized[:,:] / 255.
#load  images data and classes id
def LoadImageData(path):
    xs = []
    ys = []
    #for ex_paths in paths:
    for index, row in df_train.iterrows():        
        img_path = path + row['Image']
        igm = LoadImage(img_path)
        xs.append(igm)
        ys.append(row['classes_id'])
        print(index)
    return np.array(xs),np.array(ys)


# In[ ]:


X_train,Y_train = LoadImageData(TRAIN_IMAGE_PATH)
print("Loaded")


# In[ ]:


print("X_train ",X_train.shape)
print("Y_train ",Y_train.shape)
print("X_train ",len(df_train))
print("y_train ",Y_train)


# In[ ]:


xs = [random.randint(0, X_train.shape[0]-1) for _ in range(9)]   
print("XS ",xs)
plot_images(X_train[xs])


# In[ ]:


X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#change the classes id to 0 1 format
Y_train = keras.utils.to_categorical(Y_train,num_classes=len(unique_classes_id_dict))

print(np.shape(X_train))
print(np.shape(Y_train))


# In[ ]:


def cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides = (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization(axis = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), strides = (1,1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(len(unique_id_classes_dict), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model
model = cnn()
history = model.fit(X_train, Y_train, epochs=200, batch_size=100, verbose=1)


# In[ ]:


plt.plot(history.history['acc'], color='green', linewidth = 2, 
         marker='o', markerfacecolor='blue', markersize=4) 
plt.title('Whale Identification CNN Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()


# In[ ]:


SAMPLE_SUBMISSION_FILE="sample_submission.csv"

def getLabel(classes):
    result = []
    for i in range(0, len(classes)):
        _class = unique_id_classes_dict.get(classes[i])
        result.append(_class)
    return result

with open(SAMPLE_SUBMISSION_FILE,"w") as f:
    test_imgs = glob("../input/test/*jpg")
    f.write("Image,Id\n")
    for image in test_imgs:
        #print(image)
        igm = LoadImage(image)
        X_test = np.array(igm)
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        Y_test = model.predict_proba(X_test,batch_size=1)
        best_predict_5 = np.argsort(Y_test)[0][::-1][:5]
        pre = getLabel(best_predict_5)
        #print(image, " ".join( pre))
        f.write("%s,%s\n" %(os.path.basename(image), " ".join( pre)))
print("csv created")

