#!/usr/bin/env python
# coding: utf-8

# # Intro
# Kaggle provides free access to NVidia K80 GPUs in kernels environments. This benchmark comparison shows that adding a to your Kernel results in a ---- speedup during training of a deep learning model. 
# 
# You are currently viewing a kernel that was run with a CPU. The timing with a GPU comes from [this kernel](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu).
# 
# The total run-time for this kernel with a GPU is 994 seconds. The total run-time for the kernel with the same code running on a CPU is 13,419 seconds. This is a 12.5X speedup (total run-time with only a CPU is 13.5X as long).
# 
# Limiting the comparison only to model training, we see a reduction from 13,378 seconds on CPU to 950 seconds with a GPU.  So the model training speed-up is a little over 13X.
# 
# The exact speed-up varies based on a number of factors including model architecture, batch-size, input pipeline complexity, etc. That said, the GPU opens up much great possibilities in Kaggle kernels. 
# 
# If you want to use these GPU's for deep learning projects, you'll likely find our [Deep Learning Course](kaggle.com/learn/deep-learning) the fastest path around to get up to speed so you can run your own projects.  We're also adding new image processing datasets to our [Datasets platform](kaggle.com/datasets) and we always have many [Competitions](kaggle.com/competitions) for you to try out new ideas using these free GPU's.  
# 
# Find more details about how to use the GPU's and about this benchmark below.
# 
# 

# # Sample Images
# 
# The dataset contains images for 29 different signs. These are the 26 letters (A through Z) plus the signs for *space*, *delete* and *nothing*. Our model will view these images and learn to classify what sign is made in each image.
# 
# Sample images below

# In[ ]:


# Imports for Deep Learning
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# ensure consistency across runs
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Imports to view data
import cv2
from glob import glob
from matplotlib import pyplot as plt
from numpy import floor
import random

def plot_three_samples(letter):
    print("Samples images for letter " + letter)
    base_path = '../input/asl_alphabet_train/asl_alphabet_train/'
    img_path = base_path + letter + '/**'
    path_contents = glob(img_path)
    
    plt.figure(figsize=(16,16))
    imgs = random.sample(path_contents, 3)
    plt.subplot(131)
    plt.imshow(cv2.imread(imgs[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(imgs[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(imgs[2]))
    return

plot_three_samples('A')


# # Data Set-Up

# In[ ]:


data_dir = "../input/asl_alphabet_train/asl_alphabet_train"
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


# # Model Specification

# In[ ]:


my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(n_classes, activation='softmax'))

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])


# # Model Fitting

# In[ ]:


my_model.fit_generator(train_generator, epochs=5, validation_data=val_generator)

