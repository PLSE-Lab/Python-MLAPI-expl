#!/usr/bin/env python
# coding: utf-8

# ## Image Augmentation and Visualization with Keras
# 
# Image augmentation is a way to generate more training image from original training image.
# This way increases the image by applying several random transformations to produce a plausible image.
# In addition, models will train many aspects of the image, which will help you create a generalized model.

# In[ ]:


import os
import sys

import numpy as np
import pandas as pd
import cv2

from PIL import Image
from matplotlib import pyplot as plt

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img


# In[ ]:


DATA_PATH = '../input/aptos2019-blindness-detection'
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')
TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')

df_train = pd.read_csv(TRAIN_LABEL_PATH)
df_train['diagnosis'] = df_train['diagnosis'].astype('str')
df_train = df_train[['id_code', 'diagnosis']]
if df_train['id_code'][0].split('.')[-1] != 'png':
    for index in range(len(df_train['id_code'])):
        df_train['id_code'][index] = df_train['id_code'][index] + '.png'
X_train = df_train


# ## Image Generator

# In[ ]:


def generator(datagen):
    return datagen.flow_from_dataframe(
        dataframe=X_train, 
        directory=TRAIN_IMG_PATH,
        x_col='id_code',
        y_col='diagnosis',
        target_size=(299, 299),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

def visualization(generator):
    fig, ax = plt.subplots(1, 5, figsize=(30,50))
    count = 0
    for X_batch, y_batch in generator:
        while count < 5:
            ax[count].imshow(X_batch[count])
            count += 1
        break

        plt.show()


# #### Arguments
# 
# - rotation_range      :  Rotation range of the image.
# 
# - width_shift_range   :  Horizontal shift range of the image.
# 
# - height_shift_range  :  vertical shift range of the image.
# 
# - brightness_range    :  Range the brightness of the image.
# 
# - zoom_range          :  Range for random zoom.
# 
# - fill_mode           :  Fills the space created when rotating, moving, and shrinking an image.
# 
# - horizontal_flip     :  When set to True, flip the image vertically with a 50% probability.
# 
# - vertical_flip       :  When set to True, flip the image horizontally with a 50% probability.
# 
# - rescale             :  The original image consists of 0-255 RGB coefficients, which are too high to effectively learn the model. Therefore, it scales it by 1/255 and converts it into 0-1 range.
# 
# [Reference: Keras Documentation](https://keras.io/preprocessing/image)

# ## 1) Original Images (No augmentation)

# In[ ]:


datagen1 = ImageDataGenerator(
    rescale=1./255
)
visualization(generator(datagen1))


# ## 2) Apply horizontal flip

# In[ ]:


datagen2 = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True
)
visualization(generator(datagen2))


# ## 3) Apply brightness

# In[ ]:


datagen3 = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.5, 1.5]
)
visualization(generator(datagen3))


# ## 4) Apply vrious

# In[ ]:


datagen4 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    brightness_range=[0.5, 1.5],
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=False
)
visualization(generator(datagen4))


# ## 5) Apply Pre-Processing of ImageNet

# In[ ]:


datagen5 = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
visualization(generator(datagen5))

