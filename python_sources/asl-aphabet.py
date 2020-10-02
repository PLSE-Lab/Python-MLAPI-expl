#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# For DEEP learning
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import cv2

# ensure consistency across runs
from numpy.random import seed
seed(1)

import os
from glob import glob
import random
base_dir = '/kaggle/input/asl-alphabet/'

img = cv2.imread(os.path.join(base_dir+'asl_alphabet_train/asl_alphabet_train/W','W2547.jpg'))
plt.imshow(img)


# Any results you write to the current directory are saved as output.


# In[ ]:


# print(os.listdir("../input"))
# print("===============================================")
# print(os.listdir("../input/asl-alphabet"))
# print("===============================================")
# print(os.listdir("../input/asl-alphabet/asl_alphabet_train"))
# print("===============================================")
# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"))
# print("===============================================")
# print(os.listdir("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/A"))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for dirname, _, filenames in os.walk('/kaggle/input'):
# #     # print(dirname,_,filename)
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


def sample_random(letter):
    base_dir = os.path.join("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train",letter)
    files = [os.path.join(base_dir,x) for x in os.listdir(base_dir)]
    imgs = random.sample(files,4)
    # print(imgs)
    
    plt.figure(figsize=(16,16))
    plt.subplot(241) # https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(242)
    plt.imshow(cv2.imread(imgs[1]))
    plt.subplot(243)
    plt.imshow(cv2.imread(imgs[2]))
    plt.subplot(244)
    plt.imshow(cv2.imread(imgs[3]))

sample_random('W')


# #### INPUT DATA ( WITHOUT USING ImageDataGenerator )

# In[ ]:


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import skimage
from skimage.transform import resize

img_size = 64
train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"
test_dir = "../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/"
def input_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for d in os.listdir(folder):
        if not d.startswith('.'):
            if d in ['A']:
                label = 0
            elif d in ['B']:
                label = 1
            elif d in ['C']:
                label = 2
            elif d in ['D']:
                label = 3
            elif d in ['E']:
                label = 4
            elif d in ['F']:
                label = 5
            elif d in ['G']:
                label = 6
            elif d in ['H']:
                label = 7
            elif d in ['I']:
                label = 8
            elif d in ['J']:
                label = 9
            elif d in ['K']:
                label = 10
            elif d in ['L']:
                label = 11
            elif d in ['M']:
                label = 12
            elif d in ['N']:
                label = 13
            elif d in ['O']:
                label = 14
            elif d in ['P']:
                label = 15
            elif d in ['Q']:
                label = 16
            elif d in ['R']:
                label = 17
            elif d in ['S']:
                label = 18
            elif d in ['T']:
                label = 19
            elif d in ['U']:
                label = 20
            elif d in ['V']:
                label = 21
            elif d in ['W']:
                label = 22
            elif d in ['X']:
                label = 23
            elif d in ['Y']:
                label = 24
            elif d in ['Z']:
                label = 25
            elif d in ['del']:
                label = 26
            elif d in ['nothing']:
                label = 27
            elif d in ['space']:
                label = 28           
            else:
                label = 29
            for file in tqdm(os.listdir(folder + d)):
                img_file = cv2.imread(folder + d + '/' + file)
                if img_file is None:
                    continue
#                 print(img_file)
#                 print("================================")
                img_file = skimage.transform.resize(img_file, (img_size, img_size))
                img = np.asarray(img_file)
                X.append(img)
                y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


# x_train, y_train = input_data(train_dir)
# #x_test, y_test= get_data(test_dir) # Too few images

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2) 

# # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
# from keras.utils.np_utils import to_categorical
# y_trainHot = to_categorical(y_train, num_classes = 30)
# y_testHot = to_categorical(y_test, num_classes = 30)


# #### INPUT DATA ( USING ImageDataGenerator )

# In[ ]:


data_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64
data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")


# #### MODEL SETUP

# In[ ]:


get_ipython().run_line_magic('pinfo', 'ImageDataGenerator')


# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3, strides=1, activation = 'relu', input_shape = target_dims) )
model.add(Conv2D(filters = 16, kernel_size = 3, strides=2, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 32, kernel_size = 3, strides=1, activation = 'relu'))
model.add(Conv2D(filters = 32, kernel_size = 3, strides=2, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 64, kernel_size = 3, strides=1, activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 3, strides=2, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


model.fit_generator(train_generator, epochs=5, validation_data=val_generator)


# #### MODEL FITTING

# In[ ]:


model.fit(x_train, y_train, epochs = 4, validation_data=(xtest,ytest), verbose=1)

